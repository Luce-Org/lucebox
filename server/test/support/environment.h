#pragma once
#include <cstdlib>

#if defined(_WIN32)
#define dflash_setenv(name, value) _putenv_s(name, value)
#define dflash_unsetenv(name) _putenv_s(name, "")
#else
#define dflash_setenv(name, value) setenv(name, value, 1)
#define dflash_unsetenv(name) unsetenv(name)
#endif
