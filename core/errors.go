package core

import "errors"

// Sentinel errors for common error conditions.
var (
	// ErrInvalidShape indicates incompatible dimensions or shapes
	ErrInvalidShape = errors.New("invalid shape")

	// ErrColumnNotFound indicates a column name was not found in a DataFrame
	ErrColumnNotFound = errors.New("column not found")

	// ErrTypeMismatch indicates incompatible types in an operation
	ErrTypeMismatch = errors.New("type mismatch")

	// ErrIndexOutOfBounds indicates an index is outside valid range
	ErrIndexOutOfBounds = errors.New("index out of bounds")

	// ErrNullValue indicates a null value was encountered where not allowed
	ErrNullValue = errors.New("null value encountered")

	// ErrKeyNotFound indicates a key was not found in an index
	ErrKeyNotFound = errors.New("key not found in index")

	// ErrEmptyDataFrame indicates an operation on an empty DataFrame
	ErrEmptyDataFrame = errors.New("empty dataframe")

	// ErrDuplicateColumn indicates duplicate column names
	ErrDuplicateColumn = errors.New("duplicate column name")

	// ErrEmptySeries indicates an operation on an empty Series
	ErrEmptySeries = errors.New("empty series")

	// ErrInvalidArgument indicates an invalid argument was provided
	ErrInvalidArgument = errors.New("invalid argument")
)
