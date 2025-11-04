# Contributing to GopherData

Thank you for your interest in contributing to GopherData! We welcome contributions from the community.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Guidelines](#coding-guidelines)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Areas for Contribution](#areas-for-contribution)

---

## Code of Conduct

Be respectful, inclusive, and professional. We're all here to build great software together.

---

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/GopherData.git
   cd GopherData
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/TIVerse/GopherData.git
   ```

---

## Development Setup

### Prerequisites

- Go 1.21 or higher
- Git
- Make (optional, for convenience)

### Install Dependencies

```bash
go mod download
```

### Run Tests

```bash
go test ./...
```

### Run Tests with Coverage

```bash
go test -v -race -coverprofile=coverage.out -covermode=atomic ./...
go tool cover -html=coverage.out
```

### Run Linter

```bash
golangci-lint run
```

---

## How to Contribute

### Reporting Bugs

Create an issue with:
- **Clear title** describing the problem
- **Go version** and OS
- **Steps to reproduce** the bug
- **Expected behavior** vs actual behavior
- **Code example** (minimal reproducible case)

### Suggesting Features

Create an issue with:
- **Clear title** describing the feature
- **Use case** explaining why it's needed
- **Proposed API** (if applicable)
- **Examples** of how it would be used

### Contributing Code

1. **Create a branch**:
   ```bash
   git checkout -b feature/my-feature
   ```

2. **Make your changes**:
   - Write clear, idiomatic Go code
   - Add tests for new functionality
   - Update documentation if needed

3. **Test your changes**:
   ```bash
   go test ./...
   go vet ./...
   ```

4. **Commit your changes**:
   ```bash
   git commit -m "feat: add new feature"
   ```
   
   Use conventional commit messages:
   - `feat:` - new feature
   - `fix:` - bug fix
   - `docs:` - documentation changes
   - `test:` - adding tests
   - `refactor:` - code refactoring
   - `perf:` - performance improvements

5. **Push to your fork**:
   ```bash
   git push origin feature/my-feature
   ```

6. **Create a Pull Request** on GitHub

---

## Coding Guidelines

### Go Style

- Follow [Effective Go](https://golang.org/doc/effective_go.html)
- Use `gofmt` for formatting
- Use meaningful variable names
- Keep functions focused and small

### Package Organization

```
package_name/
├── package.go        # Main package file
├── types.go          # Type definitions
├── operations.go     # Core operations
├── helpers.go        # Helper functions
└── package_test.go   # Tests
```

### Documentation

- Add GoDoc comments for all exported types, functions, and methods
- Use complete sentences
- Include examples for complex functionality

Example:
```go
// NewDataFrame creates a new DataFrame from the given data.
// The data map keys are column names, and values are column data.
// All columns must have the same length.
//
// Example:
//   data := map[string]any{
//       "name": []string{"Alice", "Bob"},
//       "age":  []int{25, 30},
//   }
//   df, err := NewDataFrame(data)
func NewDataFrame(data map[string]any) (*DataFrame, error) {
    // implementation
}
```

### Error Handling

- Return errors instead of panicking
- Use clear error messages
- Wrap errors with context using `fmt.Errorf`

```go
if err != nil {
    return nil, fmt.Errorf("failed to read file: %w", err)
}
```

### Naming Conventions

- **Packages**: lowercase, single word (e.g., `dataframe`, `series`)
- **Files**: snake_case (e.g., `data_frame.go`)
- **Types**: PascalCase (e.g., `DataFrame`)
- **Functions**: PascalCase (exported), camelCase (unexported)
- **Variables**: camelCase
- **Constants**: PascalCase or SCREAMING_SNAKE_CASE

---

## Testing

### Writing Tests

- Place tests in `*_test.go` files
- Use table-driven tests where appropriate
- Test both success and error cases
- Use descriptive test names

Example:
```go
func TestDataFrameSelect(t *testing.T) {
    tests := []struct {
        name    string
        df      *DataFrame
        columns []string
        want    *DataFrame
        wantErr bool
    }{
        {
            name: "select existing columns",
            df:   createTestDataFrame(),
            columns: []string{"name", "age"},
            want: createExpectedResult(),
            wantErr: false,
        },
        // more test cases...
    }
    
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            got := tt.df.Select(tt.columns...)
            if !reflect.DeepEqual(got, tt.want) {
                t.Errorf("Select() = %v, want %v", got, tt.want)
            }
        })
    }
}
```

### Benchmark Tests

```go
func BenchmarkDataFrameSelect(b *testing.B) {
    df := createLargeDataFrame()
    b.ResetTimer()
    
    for i := 0; i < b.N; i++ {
        _ = df.Select("col1", "col2")
    }
}
```

### Coverage Requirements

- Aim for >80% coverage on new code
- 100% coverage for critical paths
- Don't sacrifice code quality for coverage numbers

---

## Pull Request Process

1. **Update documentation** if you've changed APIs
2. **Add tests** for new functionality
3. **Ensure all tests pass** (`go test ./...`)
4. **Run linter** (`golangci-lint run`)
5. **Update CHANGELOG.md** with your changes
6. **Write a clear PR description**:
   - What problem does this solve?
   - How does it work?
   - Any breaking changes?
   - Screenshots/examples (if applicable)

### PR Checklist

- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] No breaking changes (or justified)
- [ ] Commit messages are clear

### Review Process

- Maintainers will review your PR
- Address review feedback
- Once approved, we'll merge your PR
- Your contribution will be in the next release!

---

## Areas for Contribution

### High Priority

1. **Additional ML Algorithms**
   - Random Forest
   - Gradient Boosting
   - SVM
   - Neural Networks (basic)

2. **More I/O Formats**
   - Parquet (with build tags)
   - Excel (with build tags)
   - HDF5
   - Avro

3. **SIMD Optimizations**
   - AVX2 for amd64
   - ARM NEON for arm64
   - Benchmark improvements

4. **Time Series**
   - DatetimeIndex improvements
   - Resampling
   - Time zone handling
   - Rolling window enhancements

### Medium Priority

5. **Additional Statistics**
   - More hypothesis tests
   - Bayesian statistics
   - Survival analysis

6. **Visualization Integration**
   - Chart generation
   - Integration with plotting libraries
   - Interactive plots

7. **Performance Improvements**
   - Parallel GroupBy
   - Faster joins for large datasets
   - Memory optimizations

### Low Priority (Nice to Have)

8. **Additional Encoders**
   - BinaryEncoder
   - HashingEncoder
   - CatBoostEncoder

9. **Documentation**
   - More tutorials
   - Video guides
   - Jupyter-style notebooks (Go notebooks)

10. **Examples**
    - Industry-specific examples
    - Competition solutions
    - Case studies

---

## Questions?

- Open an issue for discussion
- Join our [GitHub Discussions](https://github.com/TIVerse/GopherData/discussions)
- Email: eshanized@tiverse.com

---

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Acknowledged in the project

Thank you for making GopherData better! 🎉

---

**Last Updated:** 2025-11-04  
**Maintainers:** @eshanized, @abhineeshpriyam, @vedanthq
