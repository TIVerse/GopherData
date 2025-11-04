package series

import (
	"math"
	"sort"

	"github.com/TIVerse/GopherData/core"
)

// Sum returns the sum of all non-null values in a numeric Series.
// Returns the zero value of T if all values are null.
func Sum[T core.NumericType](s *Series[T]) T {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var sum T
	for i := 0; i < len(s.data); i++ {
		if s.nullMask == nil || !s.nullMask.Test(i) {
			sum += s.data[i]
		}
	}
	return sum
}

// Mean returns the arithmetic mean of all non-null values in a numeric Series.
// Returns NaN if all values are null.
func Mean[T core.NumericType](s *Series[T]) float64 {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var sum float64
	count := 0

	for i := 0; i < len(s.data); i++ {
		if s.nullMask == nil || !s.nullMask.Test(i) {
			sum += float64(s.data[i])
			count++
		}
	}

	if count == 0 {
		return math.NaN()
	}

	return sum / float64(count)
}

// Std returns the sample standard deviation of all non-null values.
// Uses Bessel's correction (divides by n-1).
// Returns NaN if count < 2.
func Std[T core.NumericType](s *Series[T]) float64 {
	variance := Var(s)
	return math.Sqrt(variance)
}

// Var returns the sample variance of all non-null values.
// Uses Bessel's correction (divides by n-1).
// Returns NaN if count < 2.
func Var[T core.NumericType](s *Series[T]) float64 {
	s.mu.RLock()
	defer s.mu.RUnlock()

	// First pass: calculate mean
	var sum float64
	count := 0

	for i := 0; i < len(s.data); i++ {
		if s.nullMask == nil || !s.nullMask.Test(i) {
			sum += float64(s.data[i])
			count++
		}
	}

	if count < 2 {
		return math.NaN()
	}

	mean := sum / float64(count)

	// Second pass: calculate variance
	var variance float64
	for i := 0; i < len(s.data); i++ {
		if s.nullMask == nil || !s.nullMask.Test(i) {
			diff := float64(s.data[i]) - mean
			variance += diff * diff
		}
	}

	return variance / float64(count-1) // Bessel's correction
}

// Min returns the minimum value and true, or the zero value and false if all values are null.
func Min[T core.Comparable](s *Series[T]) (T, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var minVal T
	found := false

	for i := 0; i < len(s.data); i++ {
		if s.nullMask != nil && s.nullMask.Test(i) {
			continue
		}

		if !found {
			minVal = s.data[i]
			found = true
		} else {
			// For comparable types, we need to compare
			// This is a simplified version - in production, you'd use generics constraints
			if compare(s.data[i], minVal) < 0 {
				minVal = s.data[i]
			}
		}
	}

	return minVal, found
}

// Max returns the maximum value and true, or the zero value and false if all values are null.
func Max[T core.Comparable](s *Series[T]) (T, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var maxVal T
	found := false

	for i := 0; i < len(s.data); i++ {
		if s.nullMask != nil && s.nullMask.Test(i) {
			continue
		}

		if !found {
			maxVal = s.data[i]
			found = true
		} else {
			if compare(s.data[i], maxVal) > 0 {
				maxVal = s.data[i]
			}
		}
	}

	return maxVal, found
}

// Median returns the median of all non-null values.
// Returns NaN if all values are null.
func Median[T core.NumericType](s *Series[T]) float64 {
	s.mu.RLock()
	defer s.mu.RUnlock()

	// Collect non-null values
	values := make([]float64, 0, len(s.data))
	for i := 0; i < len(s.data); i++ {
		if s.nullMask == nil || !s.nullMask.Test(i) {
			values = append(values, float64(s.data[i]))
		}
	}

	if len(values) == 0 {
		return math.NaN()
	}

	// Sort values
	sort.Float64s(values)

	n := len(values)
	if n%2 == 0 {
		// Even number of elements: average of middle two
		return (values[n/2-1] + values[n/2]) / 2.0
	}
	// Odd number of elements: middle element
	return values[n/2]
}

// Quantile returns the value at the given quantile (0.0 to 1.0).
// Uses linear interpolation between data points.
// Returns NaN if all values are null or q is out of range.
func Quantile[T core.NumericType](s *Series[T], q float64) float64 {
	if q < 0.0 || q > 1.0 {
		return math.NaN()
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	// Collect non-null values
	values := make([]float64, 0, len(s.data))
	for i := 0; i < len(s.data); i++ {
		if s.nullMask == nil || !s.nullMask.Test(i) {
			values = append(values, float64(s.data[i]))
		}
	}

	if len(values) == 0 {
		return math.NaN()
	}

	// Sort values
	sort.Float64s(values)

	n := len(values)
	pos := q * float64(n-1)
	lower := int(pos)
	upper := lower + 1

	if upper >= n {
		return values[n-1]
	}

	// Linear interpolation
	fraction := pos - float64(lower)
	return values[lower]*(1-fraction) + values[upper]*fraction
}

// compare is a helper function to compare comparable values.
// Returns -1 if a < b, 0 if a == b, 1 if a > b.
// This is a simplified version - production code would use type switches or constraints.
func compare[T core.Comparable](a, b T) int {
	// This is a workaround since we can't directly compare generic Comparable types
	// In production, you'd use type parameters with more specific constraints
	switch any(a).(type) {
	case int, int8, int16, int32, int64:
		aInt := any(a).(int64)
		bInt := any(b).(int64)
		if aInt < bInt {
			return -1
		} else if aInt > bInt {
			return 1
		}
		return 0
	case uint, uint8, uint16, uint32, uint64:
		aUint := any(a).(uint64)
		bUint := any(b).(uint64)
		if aUint < bUint {
			return -1
		} else if aUint > bUint {
			return 1
		}
		return 0
	case float32, float64:
		aFloat := any(a).(float64)
		bFloat := any(b).(float64)
		if aFloat < bFloat {
			return -1
		} else if aFloat > bFloat {
			return 1
		}
		return 0
	case string:
		aStr := any(a).(string)
		bStr := any(b).(string)
		if aStr < bStr {
			return -1
		} else if aStr > bStr {
			return 1
		}
		return 0
	default:
		// Fallback: equal
		return 0
	}
}
