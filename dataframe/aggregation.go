package dataframe

import (
	"fmt"
	"math"

	"github.com/TIVerse/GopherData/core"
	"github.com/TIVerse/GopherData/series"
)

// AggOptions configures aggregation behavior.
type AggOptions struct {
	skipNA bool
}

// AggOption is a functional option for aggregations.
type AggOption func(*AggOptions)

// SkipNA sets whether to skip null values (default: true).
func SkipNA(skip bool) AggOption {
	return func(opts *AggOptions) {
		opts.skipNA = skip
	}
}

// Sum calculates the sum of numeric columns.
// Returns a map of column names to their sums.
func (df *DataFrame) Sum(cols ...string) (map[string]float64, error) {
	df.mu.RLock()
	defer df.mu.RUnlock()

	if len(cols) == 0 {
		// Sum all numeric columns
		cols = df.getNumericColumns()
	}

	result := make(map[string]float64)

	for _, col := range cols {
		s, exists := df.series[col]
		if !exists {
			return nil, fmt.Errorf("column %q: %w", col, core.ErrColumnNotFound)
		}

		if !isNumericType(s.Dtype()) {
			return nil, fmt.Errorf("column %q: cannot sum non-numeric type %s", col, s.Dtype())
		}

		sum := sumColumn(s)
		result[col] = sum
	}

	return result, nil
}

// Mean calculates the arithmetic mean of numeric columns.
func (df *DataFrame) Mean(cols ...string) (map[string]float64, error) {
	df.mu.RLock()
	defer df.mu.RUnlock()

	if len(cols) == 0 {
		cols = df.getNumericColumns()
	}

	result := make(map[string]float64)

	for _, col := range cols {
		s, exists := df.series[col]
		if !exists {
			return nil, fmt.Errorf("column %q: %w", col, core.ErrColumnNotFound)
		}

		if !isNumericType(s.Dtype()) {
			return nil, fmt.Errorf("column %q: cannot compute mean of non-numeric type %s", col, s.Dtype())
		}

		mean := meanColumn(s)
		result[col] = mean
	}

	return result, nil
}

// Median calculates the median of numeric columns.
func (df *DataFrame) Median(cols ...string) (map[string]float64, error) {
	df.mu.RLock()
	defer df.mu.RUnlock()

	if len(cols) == 0 {
		cols = df.getNumericColumns()
	}

	result := make(map[string]float64)

	for _, col := range cols {
		s, exists := df.series[col]
		if !exists {
			return nil, fmt.Errorf("column %q: %w", col, core.ErrColumnNotFound)
		}

		if !isNumericType(s.Dtype()) {
			return nil, fmt.Errorf("column %q: cannot compute median of non-numeric type %s", col, s.Dtype())
		}

		median := medianColumn(s)
		result[col] = median
	}

	return result, nil
}

// Std calculates the sample standard deviation of numeric columns.
func (df *DataFrame) Std(cols ...string) (map[string]float64, error) {
	df.mu.RLock()
	defer df.mu.RUnlock()

	if len(cols) == 0 {
		cols = df.getNumericColumns()
	}

	result := make(map[string]float64)

	for _, col := range cols {
		s, exists := df.series[col]
		if !exists {
			return nil, fmt.Errorf("column %q: %w", col, core.ErrColumnNotFound)
		}

		if !isNumericType(s.Dtype()) {
			return nil, fmt.Errorf("column %q: cannot compute std of non-numeric type %s", col, s.Dtype())
		}

		std := stdColumn(s)
		result[col] = std
	}

	return result, nil
}

// Var calculates the sample variance of numeric columns.
func (df *DataFrame) Var(cols ...string) (map[string]float64, error) {
	df.mu.RLock()
	defer df.mu.RUnlock()

	if len(cols) == 0 {
		cols = df.getNumericColumns()
	}

	result := make(map[string]float64)

	for _, col := range cols {
		s, exists := df.series[col]
		if !exists {
			return nil, fmt.Errorf("column %q: %w", col, core.ErrColumnNotFound)
		}

		if !isNumericType(s.Dtype()) {
			return nil, fmt.Errorf("column %q: cannot compute variance of non-numeric type %s", col, s.Dtype())
		}

		variance := varColumn(s)
		result[col] = variance
	}

	return result, nil
}

// Min returns the minimum value for each column.
func (df *DataFrame) Min(cols ...string) (map[string]any, error) {
	df.mu.RLock()
	defer df.mu.RUnlock()

	if len(cols) == 0 {
		cols = df.columns
	}

	result := make(map[string]any)

	for _, col := range cols {
		s, exists := df.series[col]
		if !exists {
			return nil, fmt.Errorf("column %q: %w", col, core.ErrColumnNotFound)
		}

		min := minColumn(s)
		result[col] = min
	}

	return result, nil
}

// Max returns the maximum value for each column.
func (df *DataFrame) Max(cols ...string) (map[string]any, error) {
	df.mu.RLock()
	defer df.mu.RUnlock()

	if len(cols) == 0 {
		cols = df.columns
	}

	result := make(map[string]any)

	for _, col := range cols {
		s, exists := df.series[col]
		if !exists {
			return nil, fmt.Errorf("column %q: %w", col, core.ErrColumnNotFound)
		}

		max := maxColumn(s)
		result[col] = max
	}

	return result, nil
}

// Count returns the count of non-null values for each column.
func (df *DataFrame) Count(cols ...string) (map[string]int, error) {
	df.mu.RLock()
	defer df.mu.RUnlock()

	if len(cols) == 0 {
		cols = df.columns
	}

	result := make(map[string]int)

	for _, col := range cols {
		s, exists := df.series[col]
		if !exists {
			return nil, fmt.Errorf("column %q: %w", col, core.ErrColumnNotFound)
		}

		count := s.Len() - s.NullCount()
		result[col] = count
	}

	return result, nil
}

// Describe generates descriptive statistics for numeric columns.
// Returns a DataFrame with statistics: count, mean, std, min, 25%, 50%, 75%, max.
func (df *DataFrame) Describe() (*DataFrame, error) {
	df.mu.RLock()
	numericCols := df.getNumericColumns()
	df.mu.RUnlock()

	if len(numericCols) == 0 {
		return nil, fmt.Errorf("no numeric columns to describe")
	}

	// Statistics to compute
	statNames := []string{"count", "mean", "std", "min", "25%", "50%", "75%", "max"}
	statsData := make(map[string][]any)

	// Initialize stat rows
	for _, stat := range statNames {
		statsData[stat] = make([]any, len(numericCols))
	}

	// Compute statistics for each column
	for i, col := range numericCols {
		s, _ := df.Column(col)

		count := float64(s.Len() - s.NullCount())
		statsData["count"][i] = count

		if count == 0 {
			// All nulls
			for j := 1; j < len(statNames); j++ {
				statsData[statNames[j]][i] = math.NaN()
			}
			continue
		}

		statsData["mean"][i] = meanColumn(s)
		statsData["std"][i] = stdColumn(s)
		statsData["min"][i] = minColumn(s)
		statsData["25%"][i] = quantileColumn(s, 0.25)
		statsData["50%"][i] = medianColumn(s)
		statsData["75%"][i] = quantileColumn(s, 0.75)
		statsData["max"][i] = maxColumn(s)
	}

	// Create result DataFrame
	// Convert map[string][]any to map[string]any for New()
	convertedData := make(map[string]any, len(statsData))
	for key, val := range statsData {
		convertedData[key] = val
	}

	result, err := New(convertedData)
	if err != nil {
		return nil, err
	}

	// Set index to stat names
	_ = result.SetIndex(NewStringIndex(statNames))

	return result, nil
}

// Helper functions

func (df *DataFrame) getNumericColumns() []string {
	numericCols := make([]string, 0)
	for _, col := range df.columns {
		s := df.series[col]
		if isNumericType(s.Dtype()) {
			numericCols = append(numericCols, col)
		}
	}
	return numericCols
}

func isNumericType(dtype core.Dtype) bool {
	return dtype == core.DtypeInt64 || dtype == core.DtypeFloat64
}

func sumColumn(s *series.Series[any]) float64 {
	var sum float64
	for i := 0; i < s.Len(); i++ {
		if s.IsNull(i) {
			continue
		}
		val := s.GetUnsafe(i)
		sum += toFloat64(val)
	}
	return sum
}

func meanColumn(s *series.Series[any]) float64 {
	var sum float64
	count := 0

	for i := 0; i < s.Len(); i++ {
		if s.IsNull(i) {
			continue
		}
		val := s.GetUnsafe(i)
		sum += toFloat64(val)
		count++
	}

	if count == 0 {
		return math.NaN()
	}

	return sum / float64(count)
}

func medianColumn(s *series.Series[any]) float64 {
	return quantileColumn(s, 0.5)
}

func stdColumn(s *series.Series[any]) float64 {
	variance := varColumn(s)
	return math.Sqrt(variance)
}

func varColumn(s *series.Series[any]) float64 {
	// Two-pass algorithm for numerical stability
	mean := meanColumn(s)
	if math.IsNaN(mean) {
		return math.NaN()
	}

	var sumSq float64
	count := 0

	for i := 0; i < s.Len(); i++ {
		if s.IsNull(i) {
			continue
		}
		val := s.GetUnsafe(i)
		diff := toFloat64(val) - mean
		sumSq += diff * diff
		count++
	}

	if count < 2 {
		return math.NaN()
	}

	return sumSq / float64(count-1) // Bessel's correction
}

func minColumn(s *series.Series[any]) any {
	var min any
	found := false

	for i := 0; i < s.Len(); i++ {
		if s.IsNull(i) {
			continue
		}
		val := s.GetUnsafe(i)
		if !found {
			min = val
			found = true
		} else {
			if compareAny(val, min) < 0 {
				min = val
			}
		}
	}

	if !found {
		return nil
	}
	return min
}

func maxColumn(s *series.Series[any]) any {
	var max any
	found := false

	for i := 0; i < s.Len(); i++ {
		if s.IsNull(i) {
			continue
		}
		val := s.GetUnsafe(i)
		if !found {
			max = val
			found = true
		} else {
			if compareAny(val, max) > 0 {
				max = val
			}
		}
	}

	if !found {
		return nil
	}
	return max
}

func quantileColumn(s *series.Series[any], q float64) float64 {
	// Collect non-null values
	values := make([]float64, 0, s.Len())
	for i := 0; i < s.Len(); i++ {
		if !s.IsNull(i) {
			val := s.GetUnsafe(i)
			values = append(values, toFloat64(val))
		}
	}

	if len(values) == 0 {
		return math.NaN()
	}

	// Simple quickselect would be more efficient, but sort is simpler for now
	sortFloat64s(values)

	pos := q * float64(len(values)-1)
	lower := int(pos)
	upper := lower + 1

	if upper >= len(values) {
		return values[len(values)-1]
	}

	// Linear interpolation
	fraction := pos - float64(lower)
	return values[lower]*(1-fraction) + values[upper]*fraction
}

func toFloat64(val any) float64 {
	switch v := val.(type) {
	case int64:
		return float64(v)
	case float64:
		return v
	case int:
		return float64(v)
	case float32:
		return float64(v)
	default:
		return 0
	}
}

func compareAny(a, b any) int {
	switch va := a.(type) {
	case int64:
		vb := b.(int64)
		if va < vb {
			return -1
		} else if va > vb {
			return 1
		}
		return 0
	case float64:
		vb := b.(float64)
		if va < vb {
			return -1
		} else if va > vb {
			return 1
		}
		return 0
	case string:
		vb := b.(string)
		if va < vb {
			return -1
		} else if va > vb {
			return 1
		}
		return 0
	case bool:
		vb := b.(bool)
		if !va && vb {
			return -1
		} else if va && !vb {
			return 1
		}
		return 0
	default:
		return 0
	}
}

func sortFloat64s(values []float64) {
	// Use quicksort for O(n log n) performance
	quickSort(values, 0, len(values)-1)
}

// quickSort implements the quicksort algorithm for float64 slices
func quickSort(arr []float64, low, high int) {
	if low < high {
		pi := partition(arr, low, high)
		quickSort(arr, low, pi-1)
		quickSort(arr, pi+1, high)
	}
}

// partition helper for quicksort
func partition(arr []float64, low, high int) int {
	pivot := arr[high]
	i := low - 1
	
	for j := low; j < high; j++ {
		if arr[j] < pivot {
			i++
			arr[i], arr[j] = arr[j], arr[i]
		}
	}
	arr[i+1], arr[high] = arr[high], arr[i+1]
	return i + 1
}
