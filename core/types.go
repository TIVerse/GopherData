// Package core provides foundational types, interfaces, and patterns for GopherData.
package core

import "fmt"

// Dtype represents the data type of a Series or DataFrame column.
type Dtype int

const (
	// DtypeInt64 represents 64-bit signed integers
	DtypeInt64 Dtype = iota
	// DtypeFloat64 represents 64-bit floating point numbers
	DtypeFloat64
	// DtypeString represents UTF-8 strings
	DtypeString
	// DtypeBool represents boolean values
	DtypeBool
	// DtypeTime represents datetime values (Phase 2)
	DtypeTime
	// DtypeCategory represents categorical/enum values (Phase 3)
	DtypeCategory
)

// String returns the string representation of a Dtype.
func (d Dtype) String() string {
	switch d {
	case DtypeInt64:
		return "int64"
	case DtypeFloat64:
		return "float64"
	case DtypeString:
		return "string"
	case DtypeBool:
		return "bool"
	case DtypeTime:
		return "datetime"
	case DtypeCategory:
		return "category"
	default:
		return "unknown"
	}
}

// NumericType is a constraint for numeric types.
type NumericType interface {
	int | int8 | int16 | int32 | int64 |
		uint | uint8 | uint16 | uint32 | uint64 |
		float32 | float64
}

// Comparable is a constraint for comparable types.
type Comparable interface {
	comparable
}

// Order represents sort ordering.
type Order int

const (
	// Ascending represents ascending sort order
	Ascending Order = iota
	// Descending represents descending sort order
	Descending
)

// String returns the string representation of an Order.
func (o Order) String() string {
	switch o {
	case Ascending:
		return "ascending"
	case Descending:
		return "descending"
	default:
		return "unknown"
	}
}

// Index provides row labels and fast label-based lookups.
// The interface is defined here in core to avoid circular dependencies.
// Implementations (RangeIndex, DatetimeIndex, StringIndex) are in dataframe/indexing.go.
type Index interface {
	// Len returns the number of elements in the index
	Len() int

	// Get returns the label at the given position
	Get(pos int) any

	// Slice returns a subset of the index from start to end (exclusive)
	Slice(start, end int) Index

	// Loc returns the integer positions for the given labels
	Loc(labels ...any) ([]int, error)

	// Copy returns a deep copy of the index
	Copy() Index
}

// String representation helper
func FormatIndex(idx Index) string {
	if idx == nil {
		return "nil"
	}
	n := idx.Len()
	if n == 0 {
		return "Index([])"
	}
	if n <= 5 {
		return fmt.Sprintf("Index(%v)", indexSlice(idx, 0, n))
	}
	return fmt.Sprintf("Index([%v ... %v], len=%d)", idx.Get(0), idx.Get(n-1), n)
}

func indexSlice(idx Index, start, end int) []any {
	result := make([]any, end-start)
	for i := start; i < end; i++ {
		result[i-start] = idx.Get(i)
	}
	return result
}
