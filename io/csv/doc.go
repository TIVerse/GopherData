// Package csv provides CSV reading and writing functionality for DataFrames.
//
// Features:
//   - Automatic type inference (int64, float64, bool, string)
//   - Configurable NA value detection
//   - Support for custom delimiters
//   - Header row handling
//   - Streaming support for large files (phase 5)
//
// Performance targets:
//   - Phase 1: >500MB/s baseline reading on NVMe SSD
//   - Phase 5: >2GB/s with parallel chunks and SIMD optimizations
//
// Example:
//
//	df, err := csv.ReadCSV("data.csv",
//	    csv.WithDelimiter(','),
//	    csv.WithHeader(true),
//	    csv.WithNA([]string{"NA", "NULL"}),
//	)
//
//	err = csv.WriteCSV(df, "output.csv")
package csv
