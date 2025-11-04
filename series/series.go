// Package series provides a generic one-dimensional labeled array.
package series

import (
	"fmt"
	"sync"

	"github.com/TIVerse/GopherData/core"
	"github.com/TIVerse/GopherData/internal/bitset"
)

// Series is a generic one-dimensional labeled array with support for null values.
// Operations return new Series (copy-on-write semantics).
type Series[T any] struct {
	name     string
	data     []T
	dtype    core.Dtype
	nullMask *bitset.BitSet // nil if no nulls present
	index    core.Index
	mu       sync.RWMutex
}

// New creates a new Series from a slice of data.
// The dtype must match the type T.
func New[T any](name string, data []T, dtype core.Dtype) *Series[T] {
	s := &Series[T]{
		name:  name,
		data:  make([]T, len(data)),
		dtype: dtype,
		index: nil, // Will be set to RangeIndex by DataFrame
	}
	copy(s.data, data)
	return s
}

// NewWithNulls creates a new Series with a pre-existing null mask.
func NewWithNulls[T any](name string, data []T, dtype core.Dtype, nullMask *bitset.BitSet) *Series[T] {
	s := &Series[T]{
		name:     name,
		data:     make([]T, len(data)),
		dtype:    dtype,
		nullMask: nullMask,
		index:    nil,
	}
	copy(s.data, data)
	return s
}

// Name returns the name of the Series.
func (s *Series[T]) Name() string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.name
}

// Len returns the number of elements in the Series.
func (s *Series[T]) Len() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.data)
}

// Dtype returns the data type of the Series.
func (s *Series[T]) Dtype() core.Dtype {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.dtype
}

// Index returns the index of the Series.
func (s *Series[T]) Index() core.Index {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.index
}

// SetIndex sets the index of the Series.
func (s *Series[T]) SetIndex(idx core.Index) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.index = idx
}

// Data returns a copy of the underlying data slice.
func (s *Series[T]) Data() []T {
	s.mu.RLock()
	defer s.mu.RUnlock()
	result := make([]T, len(s.data))
	copy(result, s.data)
	return result
}

// Get returns the value at position i and a boolean indicating if it's valid (not null).
// If the value is null, returns the zero value of T and false.
func (s *Series[T]) Get(i int) (T, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if i < 0 || i >= len(s.data) {
		var zero T
		return zero, false
	}

	if s.nullMask != nil && s.nullMask.Test(i) {
		var zero T
		return zero, false
	}

	return s.data[i], true
}

// GetUnsafe returns the value at position i without null checking.
// Use only when you're certain the value is not null.
func (s *Series[T]) GetUnsafe(i int) T {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.data[i]
}

// Set sets the value at position i and marks it as non-null.
func (s *Series[T]) Set(i int, value T) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if i < 0 || i >= len(s.data) {
		return fmt.Errorf("index %d: %w", i, core.ErrIndexOutOfBounds)
	}

	s.data[i] = value

	// Clear null bit if it exists
	if s.nullMask != nil {
		s.nullMask.Clear(i)
	}

	return nil
}

// Apply applies a function to each non-null element and returns a new Series.
func (s *Series[T]) Apply(fn func(T) T) *Series[T] {
	s.mu.RLock()
	defer s.mu.RUnlock()

	result := &Series[T]{
		name:  s.name,
		data:  make([]T, len(s.data)),
		dtype: s.dtype,
		index: s.index,
	}

	// Clone null mask if present
	if s.nullMask != nil {
		result.nullMask = s.nullMask.Clone()
	}

	for i := 0; i < len(s.data); i++ {
		if s.nullMask == nil || !s.nullMask.Test(i) {
			result.data[i] = fn(s.data[i])
		} else {
			result.data[i] = s.data[i] // Keep original (zero value)
		}
	}

	return result
}

// Filter returns a new Series containing only elements for which fn returns true.
// Null values are excluded by default.
func (s *Series[T]) Filter(fn func(T) bool) *Series[T] {
	s.mu.RLock()
	defer s.mu.RUnlock()

	filtered := make([]T, 0, len(s.data)/2) // Allocate half capacity as estimate
	var newNullMask *bitset.BitSet

	for i := 0; i < len(s.data); i++ {
		if s.nullMask != nil && s.nullMask.Test(i) {
			continue // Skip nulls
		}
		if fn(s.data[i]) {
			filtered = append(filtered, s.data[i])
		}
	}

	return &Series[T]{
		name:     s.name,
		data:     filtered,
		dtype:    s.dtype,
		nullMask: newNullMask,
		index:    nil, // Index is invalidated by filtering
	}
}

// Copy returns a deep copy of the Series.
func (s *Series[T]) Copy() *Series[T] {
	s.mu.RLock()
	defer s.mu.RUnlock()

	result := &Series[T]{
		name:  s.name,
		data:  make([]T, len(s.data)),
		dtype: s.dtype,
		index: s.index,
	}

	copy(result.data, s.data)

	if s.nullMask != nil {
		result.nullMask = s.nullMask.Clone()
	}

	return result
}

// String returns a string representation of the Series.
func (s *Series[T]) String() string {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if len(s.data) == 0 {
		return fmt.Sprintf("Series(%s): []", s.name)
	}

	const maxDisplay = 10
	n := len(s.data)

	result := fmt.Sprintf("Series(%s, dtype=%s, len=%d)\n", s.name, s.dtype, n)

	if n <= maxDisplay {
		for i := 0; i < n; i++ {
			if s.nullMask != nil && s.nullMask.Test(i) {
				result += fmt.Sprintf("  %d: <null>\n", i)
			} else {
				result += fmt.Sprintf("  %d: %v\n", i, s.data[i])
			}
		}
	} else {
		// Show first 5
		for i := 0; i < 5; i++ {
			if s.nullMask != nil && s.nullMask.Test(i) {
				result += fmt.Sprintf("  %d: <null>\n", i)
			} else {
				result += fmt.Sprintf("  %d: %v\n", i, s.data[i])
			}
		}
		result += "  ...\n"
		// Show last 5
		for i := n - 5; i < n; i++ {
			if s.nullMask != nil && s.nullMask.Test(i) {
				result += fmt.Sprintf("  %d: <null>\n", i)
			} else {
				result += fmt.Sprintf("  %d: %v\n", i, s.data[i])
			}
		}
	}

	return result
}

// Slice returns a new Series containing elements from start (inclusive) to end (exclusive).
func (s *Series[T]) Slice(start, end int) *Series[T] {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if start < 0 {
		start = 0
	}
	if end > len(s.data) {
		end = len(s.data)
	}
	if start >= end {
		return &Series[T]{
			name:  s.name,
			data:  []T{},
			dtype: s.dtype,
		}
	}

	result := &Series[T]{
		name:  s.name,
		data:  make([]T, end-start),
		dtype: s.dtype,
	}

	copy(result.data, s.data[start:end])

	if s.nullMask != nil {
		result.nullMask = s.nullMask.Slice(start, end)
	}

	return result
}
