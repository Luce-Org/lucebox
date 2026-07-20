# CppUnitTestFramework
A single header C++17 unit test framework with a focus on simplicity and quick setup.  It should compile with any C++17 compliant compiler without any additional source or binary dependencies.  Platform specific operations are avoided.

Visual Studio Code Test Adapter: [vscode-cpputf-test-adapter](https://marketplace.visualstudio.com/items?itemName=drleq.vscode-cpputf-test-adapter)  
Visual Studio Test Adapter: [CppUnitTestFrameworkTestAdapter](https://marketplace.visualstudio.com/items?itemName=drleq.CppUnitTestFrameworkTestAdapter)


# Quick start
1. Copy (or clone) the header to your local codebase.
1. Create an entry-point file with the following contents to generate a `main` function:
    ```cpp
    #define GENERATE_UNIT_TEST_MAIN
    #include "CppUnitTestFramework.hpp"
    ```
1. Create individual `.cpp` files for your test fixtures and use the `TEST_CASE()` macro to create individual test cases:
    ```cpp
    #include "CppUnitTestFramework.hpp"

    namespace {
        struct MyFixture { ... };
    }

    TEST_CASE(MyFixture, Test1) { ... }
    TEST_CASE(MyFixture, Test2) { ... }
    ```
1. Compile your program and run.


# Command-line options
The default `main` function supports a small number of command line options:
```
Usage: <program> [<options>] [keyword1] [keyword2] ...
    -h, --help, -?:        Displays this message
    -v, --verbose:         Show verbose output
        --discover_tests:  Output test details
        --adapter_info:    Output additional details for test adapters
```
If no `options` or `keywords` are provided then all test cases are run but only test failures are recorded.  The `--verbose` option will force all test cases to be recorded, even if they pass or are skipped.  Any `keywords` provided will be used to filter the set of test cases.


# Fixtures and test cases
A test fixture is a base class that is re-used for multiple test cases.  Each test case will have it's own copy of the base class so each test case will perform the same set-up and tear-down steps.
```cpp
struct MyFixture {
    constexpr int CommonData = 10;
};

TEST_CASE(MyFixture, Test1) {
    // MyFixture constructor called
    CHECK_EQUAL(CommonData, 10);
    // MyFixture destructor called
}
TEST_CASE(MyFixture, Test2) {
    // MyFixture constructor called
    CHECK_EQUAL(CommonData, 10);
    // MyFixture destructor called
}
```

If you want to use assertions in the test fixture base class then it should inherit from `CppUnitTestFramework::CommonFixture`.
```cpp
struct MyFixture : CppUnitTestFramework::CommonFixture {
    // Constructors should be forwarded to allow for initialization
    using CppUnitTestFramework::CommonFixture::CommonFixture;

    void CustomAssert(int a, int b) {
        // This will only compile if CommonFixture is a base class
        CHECK_EQUAL(a, b)
    }
};
```


# Tags and keywords
Test cases can be optionally tagged, allowing them to be grouped into categories that span multiple test files.  For example, given the following tests:
```cpp
TEST_CASE(MyFixture, TestWithoutTags) { ... }
TEST_CASE_WITH_TAGS(MyFixture, TestWithTags, "gpu") { ... }
TEST_CASE_WITH_TAGS(OtherFixture, TestWithTags, "gpu") { ... }
```
You can then execute the tests individually using `keywords`:
```bash
./MyTests               # Runs all tests
./MyTests MyFixture     # Runs both MyFixture tests
./MyTests TestWithTags  # Runs both test cases called TestWithTags
./MyTests gpu           # Runs both test cases tagged as "gpu"
```
All keyword matching is case sensitive.  Keywords will match a fixture or test case name if they occur anywhere in the name, but must match a test tag exactly.  For example:
```bash
./MyTests MyFix         # Runs both MyFixture tests
./MyTests gp            # Fails to match any tests
```

# Assertions
Assertions are provided in two flavors: `REQUIRE` and `CHECK`.  A `REQUIRE` assertion will cause a test case to immediately fail, while a `CHECK` assertion will allow the test case to continue but will still cause a failure once it completes.  Both flavors include a basic set of assertion types:
```cpp
REQUIRE(Expression)            // Same as REQUIRE_TRUE
REQUIRE_TRUE(Expression)       // Asserts that [Expression] evaluates to [true]
REQUIRE_FALSE(Expression)      // Asserts that [Expression] evaluates to [false]
REQUIRE_EQUAL(Left, Right)     // Asserts that [Left == Right]
REQUIRE_NULL(Expression)       // Asserts that [Expression] evaluates to [nullptr]
REQUIRE_NOT_NULL(Expression)   // Asserts that [Expression] does not evaluate to [nullptr]
REQUIRE_THROW(ExceptionType, Expession)  // Asserts that invoking [Expression] causes an exception of type [ExceptionType] to be thrown
REQUIRE_NO_THROW(Expression)   // Asserts that invoking [Expression] does not cause any type of exception to be thrown
REQUIRE_CLOSE(Left, Right, Percentage)   // Asserts that [abs(Right - Left)] is less than the absolute [Percentage] of [Left] or [Right]
REQUIRE_CLOSE_FRACITON(Left, Right, Fraction)  // Asserts that [abs(Right - Left)] is less than [Fraction]
```
Each of these assertion macros invoke an equivalent method in the `CppUnitTestFramework::Assert` namespace.  These methods can be overloaded in your own code if additional customization is required:
```cpp
namespace CppUnitTestFramework::Assert {
    template <typename TLeft, typename TRight>
    std::optional<AssertException> AreEqual(const TLeft& left, const TRight& right, const char* expression);
    template <typename T>
    std::optional<AssertException> IsNull(const T& value, const char* expression);
    template <typename T>
    std::optional<AssertException> IsNotNull(const T& value, const char* expression);
    std::optional<AssertException> IsTrue(bool value, const char* expression);
    std::optional<AssertException> IsFalse(bool value, const char* expression);
    template <typename TException, typename Callback>
    std::optional<AssertException> Throws(const Callback& callback);
    template <typename Callback>
    std::optional<AssertException> NoThrow(const Callback& callback);
    std::optional<AssertException> Close(float left, float right, float percentage);
    std::optional<AssertException> Close(double left, double right, double percentage);
    std::optional<AssertException> CloseFraction(float left, float right, float fraction);
    std::optional<AssertException> CloseFraction(double left, double right, double fraction);
}
```
If an assertion fails then a failure message is generated.  In the case of `REQUIRE_EQUAL` the `Left` and `Right` values are converted to a `std::string` to be included in the message.  This conversion is done through an overload of the `CppUnitTestFramework::Ext::ToString()` method.  Standard coversions are provided for `nullptr`, pointers, enums and any type that can be converted to a `std::string` by construction or `std::to_string()`.
```cpp
namespace CppUnitTestFramework::Ext {
    template <typename T>
    std::string ToString(const T& value);
}
```

# Sections and BDD
Within a test case it is possible to provide smaller scoped sections that isolate specific test functionality.  Sections can be nested as required.  The behavior of `REQUIRE` and `CHECK` assertions are unaffected by sections, but the test record will include the section text as it progresses.
```cpp
TEST_CASE(MyFixture, Test1) {
    SECTION("Construction") {
        SECTION("Default constructor") {
            // Test the default constructor here
        }
        SECTION("Copy-constructor") {
            // Test the copy-constructor here
        }
    }
}
```

There is basic support for [Behavior Driven Development](https://en.wikipedia.org/wiki/Behavior-driven_development) using special section types.  For example:
```cpp
TEST_CASE(MyFixture, Test1) {
    SCENARIO("Refunded items should be returned to stock") {
        GIVEN("a customer previously bought a black sweater from me") { ... }
        AND("I have three black sweaters in stock") { ... }
        WHEN("they return the black sweater for a refund") { ... }
        THEN("I should have four black sweaters in stock") { ... }
    }
}
```

# Utilities
A utility macro called `UNUSED_RETURN()` is provided to assist with the `[[nodiscard]]` attribute.  This is useful when calling `REQUIRE_THROW` with a method that is marked as non-discardable.
```cpp
[[nodiscard]] int Foo() { throw std::logic_error("Always fails"); }
REQUIRE_THROW(std::logic_error, UNUSED_RETURN(Foo()));
```

# Custom initialization and main()
In situations where you want to perform your own global setup and shutdown (such as initializing other libraries) you must provide your own `main()` method.  This simply involves copying the `main()` method from the end of the `CppUnitTestFramework.hpp` file:
```cpp
int main(int argc, const char* argv[]) {
    // Check the command line options
    CppUnitTestFramework::RunOptions options;
    if (!options.ParseCommandLine(argc, argv)) {
        return 2;
    }

    // Custom initialization here

    bool success = CppUnitTestFramework::TestRegistry::Run(
        &options,
        CppUnitTestFramework::ConsoleLogger::Create(&options)
    );

    // Custom shutdown here

    return success ? 0 : 1;
}
```