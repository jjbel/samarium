// SEE https://github.com/znone/call_thunk/blob/master/LICENSE for license

#include "call_thunk.hpp"

#ifdef _WIN32

#include <windows.h>

#else
#include <sys/mman.h>

#ifndef offsetof
#define offsetof(s, m) ((size_t) & reinterpret_cast<char const volatile&>((((s*)0)->m)))
#endif // offsetof

#ifndef _countof
#define _countof(_Array) (sizeof(_Array) / sizeof(_Array[0]))
#endif //_countof

#endif //_WIN32

#include <assert.h>
#include <memory>
#include <string.h>

namespace call_thunk
{

#pragma pack(push, 1)


struct thunk_code
{
    struct backup_pc
    {
        // pop qword ptr [_rip]

        short save_rip; // pop qword ptr [_rip]
        std::int32_t _rip;

        backup_pc(intptr_t& rip)
        {
            save_rip = 0x058F;
            _rip     = offset(&rip, _rip);
        }
    };

    struct restore_pc
    {
        // push _rip
        short restore_rip;
        std::int32_t _rip;

        restore_pc(intptr_t& rip)
        {
            restore_rip = (short)0x35FF;
            _rip        = offset(&rip, _rip);
        }
    };

    // Move the parameters in the stack to make it to the top of the stack
    struct alignment_stack
    {
        /*
        push rcx
        push rdi
        push rsi
        mov rcx, count
        lea rdi, [rsp+(offset+4)*8]
        lea rsi, [rsp+(offset+3)*8]
        rep movsq
        pop rsi
        pop rdi
        pop rcx
        */

        char backup_reg[3];
        char set_rcx[3];
        uint32_t _count;
        char copy_src[4];
        char _src;
        char copy_dest[4];
        char _dest;
        char move[3];
        char restore_reg[3];

        alignment_stack(char offset, uint32_t count)
        {
            memcpy(backup_reg, "\x51\x57\x56", _countof(backup_reg));
            memcpy(set_rcx, "\x48\xC7\xC1", _countof(set_rcx));
            _count = count;
            memcpy(copy_src, "\x48\x8D\x74\x24", sizeof(copy_src));
            memcpy(copy_dest, "\x48\x8D\x7C\x24", sizeof(copy_dest));
            _src  = (offset + 4) * 0x8;
            _dest = (offset + 3) * 0x8;
            memcpy(move, "\xF3\x48\xA5", _countof(move));
            memcpy(restore_reg, "\x5E\x5F\x59", _countof(restore_reg));
        }
    };

    struct alignment_stack1
    {
        /*
        mov rax, qword ptr [rsp+30h]
        mov qword ptr [rsp+28h], rax
        */

        char copy_src[4];
        char _src;
        char copy_dest[4];
        char _dest;

        alignment_stack1(char offset)
        {
            memcpy(copy_src, "\x48\x8B\x44\x24", sizeof(copy_src));
            memcpy(copy_dest, "\x48\x89\x44\x24", sizeof(copy_dest));
            _src  = (offset + 1) * 0x8;
            _dest = offset * 0x8;
        }
    };

    struct alignment_stack2
    {
        /*
        MOVDQA xmm0, xmmword ptr [rsp+30h]
        MOVDQU xmmword ptr [rsp+28h], xmm0
        */

        char copy_src[5];
        char _src;
        char copy_dest[5];
        char _dest;

        alignment_stack2(char offset)
        {
            memcpy(copy_src, "\x66\x0F\x6F\x44\x24", sizeof(copy_src));
            memcpy(copy_dest, "\xF3\x0F\x7F\x44\x24", sizeof(copy_dest));
            _src  = (offset + 1) * 0x8;
            _dest = offset * 0x8;
        }
    };

    struct adjust_params
    {
        /* MSVC
        sub rsp, 16
        mov [rsp+20h], r9	(or movsd mmword ptr [rsp+20h],xmm3)
        mov r9, r8		(or movsd xmm3, xmm2)
        mov r8, rdx		(or movsd xmm2, xmm1)
        mov rdx, rcx	(or movsd xmm1, xmm0)
        mov rcx, _this
        */
        /* GCC
        push 0
        push r9
        mov r9, r8
        mov r8, rcx
        mov rcx, rdx
        mov rdx, rsi
        mov rsi, rdi
        mov rdi, _this
        */

        char adjust_code[1];

        template <std::size_t N> inline char* copy_code(char* dest, const char (&code)[N])
        {
            memcpy(dest, code, N - 1);
            return dest + N - 1;
        }

        adjust_params(std::size_t argc, const argument_info* arginfos)
        {
            char* p = adjust_code;
#if defined(_WIN32)
            switch (argc)
            {
            default:
                p = copy_code(p, "\x48\x83\xEC\x10");
                if (arginfos && arginfos[3].as_floating())
                    p = copy_code(p, "\xF2\x0F\x11\x5C\x24\x20");
                else
                    p = copy_code(p, "\x4C\x89\x4C\x24\x20");
            case 3:
                if (arginfos && arginfos[2].as_floating()) p = copy_code(p, "\xF2\x0F\x10\xDA");
                else
                    p = copy_code(p, "\x4D\x8B\xC8");
            case 2:
                if (arginfos && arginfos[1].as_floating()) p = copy_code(p, "\xF2\x0F\x10\xD1");
                else
                    p = copy_code(p, "\x4C\x8B\xC2");
            case 1:
                if (arginfos && arginfos[0].as_floating()) p = copy_code(p, "\xF2\x0F\x10\xC8");
                else
                    p = copy_code(p, "\x48\x8B\xD1");
            case 0:
                if (argc == 5)
                {
                    alignment_stack1* _alignment_stack = reinterpret_cast<alignment_stack1*>(p);
                    new (_alignment_stack) alignment_stack1(5);
                    p += sizeof(alignment_stack1);
                }
                else if (argc == 6)
                {
                    alignment_stack2* _alignment_stack = reinterpret_cast<alignment_stack2*>(p);
                    new (_alignment_stack) alignment_stack2(5);
                    p += sizeof(alignment_stack2);
                }
                else if (argc > 6)
                {
                    alignment_stack* _alignment_stack = reinterpret_cast<alignment_stack*>(p);

// TODO size_t to uint32_t may lose data
#pragma warning(suppress : 4267)
                    new (_alignment_stack) alignment_stack(5, argc - 4);
                    p += sizeof(alignment_stack);
                }
                p = copy_code(p, "\x48\xB9");
            }

#else
            std::size_t iargc = 0, fargc = 0, sargc = 0;
            if (arginfos)
            {
                for (std::size_t i = 0; i != argc; i++)
                {
                    if (arginfos[i].as_floating()) ++fargc;
                    else
                        ++iargc;
                }
            }
            else { iargc = argc; }
            switch (iargc)
            {
            default: p = copy_code(p, "\x6A\x00\x41\x51");
            case 5: p = copy_code(p, "\x4D\x8B\xC8");
            case 4: p = copy_code(p, "\x4C\x8B\xC1");
            case 3: p = copy_code(p, "\x48\x8B\xCA");
            case 2: p = copy_code(p, "\x48\x8B\xD6");
            case 1: p = copy_code(p, "\x48\x8B\xF7");
            case 0:
                if (iargc > 6)
                {
                    sargc = (iargc - 6) + std::max(int(fargc - 8), 0);
                    if (sargc == 1)
                    {
                        alignment_stack1* _alignment_stack = reinterpret_cast<alignment_stack1*>(p);
                        new (_alignment_stack) alignment_stack1(1);
                        p += sizeof(alignment_stack1);
                    }
                    else if (sargc > 1)
                    {
                        alignment_stack* _alignment_stack = reinterpret_cast<alignment_stack*>(p);
                        new (_alignment_stack) alignment_stack(1, sargc);
                        p += sizeof(alignment_stack);
                    }
                }
                p = copy_code(p, "\x48\xBF");
            }
#endif
        }

        static std::size_t
        calc_size(std::size_t argc, const argument_info* arginfos, bool* push_param_to_stack)
        {
            std::size_t n        = 0;
            *push_param_to_stack = false;
#if defined(_WIN32)
            if (argc > 3)
            {
                n += 4;
                if (argc > 4) *push_param_to_stack = true;
                if (argc == 5) n += sizeof(alignment_stack1);
                else if (argc == 6)
                    n += sizeof(alignment_stack2);
                else if (argc > 6)
                    n += sizeof(alignment_stack);
                n += (arginfos && arginfos[3].as_floating()) ? 6 : 5;
                argc = 3;
            }
            for (std::size_t i = 0; i != argc; i++)
                n += (arginfos && arginfos[i].as_floating()) ? 4 : 3;
#else
            std::size_t iargc = 0, fargc = 0;
            if (arginfos)
            {
                for (std::size_t i = 0; i != argc; i++)
                    if (arginfos[i].as_floating()) ++fargc;
                    else
                        ++iargc;
            }
            else { iargc = argc; }
            if (iargc > 5)
            {
                n += 4;
                if (argc > 6) *push_param_to_stack = true;
                std::size_t sargc = (iargc - 6) + std::max(int(fargc - 8), 0);
                if (sargc == 1) n += sizeof(alignment_stack1);
                else if (argc > 1)
                    n += sizeof(alignment_stack);
                iargc = 5;
            }
            n += iargc * 3;
#endif
            n += 2 + sizeof(intptr_t);

            return n;
        }
    };

    struct jump_function
    {
        /*
        mov rax, _proc
        jmp rax
        */

        short mov_proc; // mov rax, _proc
        intptr_t _proc;
        short call_proc; // call rax

        jump_function()
        {

// TODO cast truncates constant value
#pragma warning(suppress : 4310)
            mov_proc  = (short)0xB848;
#pragma warning(suppress : 4310)
            call_proc = (short)0xE0FF;
        }
    };

    struct call_function
    {
        /*
        mov rax, _proc
        call rax
        */

        short mov_proc; // mov rax, _proc
        intptr_t _proc;
        short call_proc; // call rax

        call_function()
        {
            
// TODO cast truncates constant value
#pragma warning(suppress : 4310)
            mov_proc  = (short)0xB848;
#pragma warning(suppress : 4310)
            call_proc = (short)0xD0FF;
        }
    };

    struct return_caller
    {
        /*
        ret n
        */
        char ret;
        short _n;

        return_caller(short n)
        {
            
// TODO truncation of constant value
#pragma warning(suppress : 4309)
            ret = 0xC2;
            _n  = n * 8;
        }
    };

    adjust_params* _adjust_params;
    jump_function* _jump_function;
    call_function* _call_function;
    restore_pc* _restore_pc;

    intptr_t _rip;

    thunk_code(call_declare, call_declare, std::size_t argc, const argument_info* arginfos)
    {
        memset(this, 0, sizeof(thunk_code));

        char* code               = reinterpret_cast<char*>(this + 1);
        bool push_param_to_stack = false;
        backup_pc* _backup_pc    = reinterpret_cast<backup_pc*>(code);
        new (_backup_pc) backup_pc(_rip);
        _adjust_params = reinterpret_cast<adjust_params*>(_backup_pc + 1);
        new (_adjust_params) adjust_params(argc, arginfos);

        code = reinterpret_cast<char*>(_adjust_params) +
               adjust_params::calc_size(argc, arginfos, &push_param_to_stack);
        if (push_param_to_stack)
        {
            _call_function = reinterpret_cast<call_function*>(code);
            new (_call_function) call_function();
            _restore_pc = reinterpret_cast<restore_pc*>(_call_function + 1);
            new (_restore_pc) restore_pc(_rip);
            return_caller* _return_caller = reinterpret_cast<return_caller*>(_restore_pc + 1);
            new (_return_caller) return_caller(2);
        }
        else
        {
            _restore_pc = reinterpret_cast<restore_pc*>(code);
            new (_restore_pc) restore_pc(_rip);
            _jump_function = reinterpret_cast<jump_function*>(_restore_pc + 1);
            new (_jump_function) jump_function();
        }
    }

    void bind(void* object, void* proc)
    {
        if (_jump_function)
        {
            intptr_t* _this       = reinterpret_cast<intptr_t*>(_restore_pc) - 1;
            *_this                = reinterpret_cast<intptr_t>(object);
            _jump_function->_proc = (intptr_t)proc;
        }
        else if (_call_function)
        {
            intptr_t* _this       = reinterpret_cast<intptr_t*>(_call_function) - 1;
            *_this                = reinterpret_cast<intptr_t>(object);
            _call_function->_proc = (intptr_t)proc;
        }
    }

    static std::int32_t offset(const void* var, std::int32_t& code)
    {
        return static_cast<std::int32_t>((const char*)var - (char*)(&code + 1));
    }

    static std::size_t
    calc_size(call_declare, call_declare, std::size_t argc, const argument_info* arginfos)
    {
        bool push_param_to_stack = false;
        std::size_t n            = adjust_params::calc_size(argc, arginfos, &push_param_to_stack);
        n                        = sizeof(thunk_code) + sizeof(backup_pc) + n + sizeof(restore_pc);
        if (push_param_to_stack) n += sizeof(call_function) + sizeof(return_caller);
        else
            n += sizeof(jump_function);
        return n;
    }
};

void base_thunk::init_code(call_declare caller,
                           call_declare callee,
                           size_t argc,
                           const argument_info* arginfos) /* CHANGED: throw(bad_call) */
{
    _thunk_size = thunk_code::calc_size(caller, callee, argc, arginfos);

#if defined(_WIN32)
    _code = (char*)VirtualAlloc(NULL, _thunk_size, MEM_COMMIT, PAGE_EXECUTE_READWRITE);
#else
    _code = (char*)mmap(NULL, _thunk_size, PROT_READ | PROT_WRITE | PROT_EXEC,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#endif //_WIN32
    _thunk = reinterpret_cast<thunk_code*>(_code);
    _code += sizeof(thunk_code);
    new (_thunk) thunk_code(caller, callee, argc, arginfos);
}

void base_thunk::destroy_code()
{
    if (_thunk)
    {
#if defined(_WIN32)
        VirtualFree(_thunk, 0, MEM_RELEASE);
#else
        munmap(_thunk, _thunk_size);
#endif //_WIN32
        _thunk      = NULL;
        _code       = NULL;
        _thunk_size = 0;
    }
}

void base_thunk::flush_cache()
{
#ifdef _WIN32
    FlushInstructionCache(GetCurrentProcess(), _thunk, _thunk_size);
#else
#endif //_WIN32
}

void base_thunk::bind_impl(void* object, void* proc) { _thunk->bind(object, proc); }

} // namespace call_thunk
