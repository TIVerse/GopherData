// Package dataframe provides a two-dimensional labeled data structure.
//
// DataFrame is the primary data structure in GopherData, analogous to pandas.DataFrame
// in Python or data.frame in R. It consists of:
//   - Multiple Series (columns) with potentially different types
//   - A shared Index for row labels
//   - Copy-on-write semantics for predictable behavior
//
// Key features:
//   - Type-safe generic Series under the hood
//   - Efficient null handling via bit-packed masks
//   - Thread-safe concurrent reads
//   - Rich selection and filtering operations
//   - Automatic parallelism for expensive operations
//
// Example:
//
//	df, err := dataframe.New(map[string]any{
//	    "name": []string{"Alice", "Bob", "Charlie"},
//	    "age":  []int64{25, 30, 35},
//	})
//
//	subset := df.Select("name", "age")
//	filtered := subset.Filter(func(r *Row) bool {
//	    age, _ := r.Get("age")
//	    return age.(int64) > 25
//	})
package dataframe
