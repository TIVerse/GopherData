// Package core provides foundational types, interfaces, and patterns used throughout GopherData.
//
// This package defines:
//   - Core type system (Dtype enum)
//   - Generic type constraints (NumericType, Comparable)
//   - Foundational interfaces (Index, Iterator, Aggregator)
//   - Sentinel errors
//   - Functional options pattern
//   - Global constants and defaults
//
// The core package has no dependencies on other GopherData packages and can be imported
// by all modules to avoid circular dependencies.
package core
