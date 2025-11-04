// Package series provides a generic one-dimensional labeled array with null support.
//
// Series[T] is the fundamental building block for DataFrames. Each DataFrame column
// is internally represented as a Series.
//
// Key features:
//   - Generic type-safe implementation using Go generics
//   - Bit-packed null masks for memory efficiency
//   - Copy-on-write semantics for predictability
//   - Thread-safe concurrent reads
//   - Rich operations: Apply, Filter, aggregations
//
// Null handling:
//   - Nulls are tracked via a separate bit mask (1 bit per value)
//   - Operations skip nulls by default
//   - FillNA and DropNA methods for null handling
//
// Example:
//
//	ages := series.New("age", []int64{25, 30, 35}, core.DtypeInt64)
//	ages.SetNull(1) // Mark index 1 as null
//	mean := series.Mean(ages) // Skips null, returns (25+35)/2 = 30
package series
