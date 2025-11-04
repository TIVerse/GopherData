package core

// Iterator provides a generic iterator interface for traversing sequences.
type Iterator[T any] interface {
	// Next advances the iterator and returns true if a value is available
	Next() bool

	// Value returns the current value
	Value() T

	// Error returns any error that occurred during iteration
	Error() error
}

// Aggregator defines the interface for custom aggregation functions.
type Aggregator interface {
	// Aggregate computes an aggregation over a slice of values
	Aggregate(values []any) (any, error)

	// Name returns the name of this aggregation function
	Name() string
}
