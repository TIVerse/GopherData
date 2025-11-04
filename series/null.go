package series

import (
	"github.com/TIVerse/GopherData/internal/bitset"
)

// IsNull returns true if the value at position i is null.
func (s *Series[T]) IsNull(i int) bool {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if i < 0 || i >= len(s.data) {
		return false
	}

	return s.nullMask != nil && s.nullMask.Test(i)
}

// SetNull marks the value at position i as null.
func (s *Series[T]) SetNull(i int) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if i < 0 || i >= len(s.data) {
		return
	}

	// Create null mask if it doesn't exist
	if s.nullMask == nil {
		s.nullMask = bitset.New(len(s.data))
	}

	s.nullMask.Set(i)
}

// NullCount returns the number of null values in the Series.
func (s *Series[T]) NullCount() int {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.nullMask == nil {
		return 0
	}

	return s.nullMask.Count()
}

// HasNulls returns true if the Series contains any null values.
func (s *Series[T]) HasNulls() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()

	return s.nullMask != nil && s.nullMask.Any()
}

// DropNA returns a new Series with null values removed.
func (s *Series[T]) DropNA() *Series[T] {
	s.mu.RLock()
	defer s.mu.RUnlock()

	// If no nulls, return a copy
	if s.nullMask == nil || s.nullMask.None() {
		return s.Copy()
	}

	// Count non-null values
	nonNullCount := len(s.data) - s.nullMask.Count()
	result := &Series[T]{
		name:     s.name,
		data:     make([]T, 0, nonNullCount),
		dtype:    s.dtype,
		nullMask: nil, // No nulls in result
	}

	for i := 0; i < len(s.data); i++ {
		if !s.nullMask.Test(i) {
			result.data = append(result.data, s.data[i])
		}
	}

	return result
}

// FillNA returns a new Series with null values replaced by the given value.
func (s *Series[T]) FillNA(value T) *Series[T] {
	s.mu.RLock()
	defer s.mu.RUnlock()

	// If no nulls, return a copy
	if s.nullMask == nil || s.nullMask.None() {
		return s.Copy()
	}

	result := &Series[T]{
		name:     s.name,
		data:     make([]T, len(s.data)),
		dtype:    s.dtype,
		nullMask: nil, // All nulls are filled
	}

	for i := 0; i < len(s.data); i++ {
		if s.nullMask.Test(i) {
			result.data[i] = value
		} else {
			result.data[i] = s.data[i]
		}
	}

	return result
}

// NullMask returns a copy of the null mask, or nil if no nulls exist.
func (s *Series[T]) NullMask() *bitset.BitSet {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.nullMask == nil {
		return nil
	}

	return s.nullMask.Clone()
}
