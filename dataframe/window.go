package dataframe

import (
	"fmt"
	"math"

	"github.com/TIVerse/GopherData/core"
	"github.com/TIVerse/GopherData/series"
)

// WindowOptions configures window behavior.
type WindowOptions struct {
	minPeriods int
	center     bool
}

// WindowOption is a functional option for windows.
type WindowOption func(*WindowOptions)

// MinPeriods sets the minimum number of observations in window.
func MinPeriods(n int) WindowOption {
	return func(opts *WindowOptions) {
		opts.minPeriods = n
	}
}

// Center centers the window around the current value.
func Center() WindowOption {
	return func(opts *WindowOptions) {
		opts.center = true
	}
}

// Window represents a rolling window over a DataFrame.
type Window struct {
	df         *DataFrame
	size       int
	minPeriods int
	center     bool
	windowType string // "rolling", "expanding", "ewm"
	alpha      float64 // For EWM
}

// Rolling creates a rolling window.
func (df *DataFrame) Rolling(size int, opts ...WindowOption) *Window {
	winOpts := &WindowOptions{
		minPeriods: size,
		center:     false,
	}
	for _, opt := range opts {
		opt(winOpts)
	}

	return &Window{
		df:         df,
		size:       size,
		minPeriods: winOpts.minPeriods,
		center:     winOpts.center,
		windowType: "rolling",
	}
}

// Expanding creates an expanding window.
func (df *DataFrame) Expanding(minPeriods int) *Window {
	return &Window{
		df:         df,
		size:       -1, // Expanding has no fixed size
		minPeriods: minPeriods,
		center:     false,
		windowType: "expanding",
	}
}

// EWM creates an exponentially weighted moving window.
func (df *DataFrame) EWM(alpha float64) *Window {
	if alpha <= 0 || alpha >= 1 {
		alpha = 0.5 // Default
	}

	return &Window{
		df:         df,
		size:       -1,
		minPeriods: 1,
		center:     false,
		windowType: "ewm",
		alpha:      alpha,
	}
}

// Mean calculates the rolling mean for a column.
func (w *Window) Mean(col string) (*series.Series[float64], error) {
	if !w.df.HasColumn(col) {
		return nil, fmt.Errorf("column %q: %w", col, core.ErrColumnNotFound)
	}

	s, _ := w.df.Column(col)
	if !isNumericType(s.Dtype()) {
		return nil, fmt.Errorf("column %q: cannot compute mean of non-numeric type", col)
	}

	nrows := w.df.Nrows()
	result := make([]float64, nrows)

	for i := 0; i < nrows; i++ {
		windowStart, windowEnd := w.getWindowBounds(i, nrows)
		values := w.extractWindowValues(s, windowStart, windowEnd)

		if len(values) < w.minPeriods {
			result[i] = math.NaN()
		} else {
			if w.windowType == "ewm" {
				result[i] = w.ewmMean(values)
			} else {
				result[i] = windowMean(values)
			}
		}
	}

	return series.New(col+"_mean", result, core.DtypeFloat64), nil
}

// Sum calculates the rolling sum for a column.
func (w *Window) Sum(col string) (*series.Series[float64], error) {
	if !w.df.HasColumn(col) {
		return nil, fmt.Errorf("column %q: %w", col, core.ErrColumnNotFound)
	}

	s, _ := w.df.Column(col)
	if !isNumericType(s.Dtype()) {
		return nil, fmt.Errorf("column %q: cannot compute sum of non-numeric type", col)
	}

	nrows := w.df.Nrows()
	result := make([]float64, nrows)

	for i := 0; i < nrows; i++ {
		windowStart, windowEnd := w.getWindowBounds(i, nrows)
		values := w.extractWindowValues(s, windowStart, windowEnd)

		if len(values) < w.minPeriods {
			result[i] = math.NaN()
		} else {
			result[i] = windowSum(values)
		}
	}

	return series.New(col+"_sum", result, core.DtypeFloat64), nil
}

// Std calculates the rolling standard deviation for a column.
func (w *Window) Std(col string) (*series.Series[float64], error) {
	if !w.df.HasColumn(col) {
		return nil, fmt.Errorf("column %q: %w", col, core.ErrColumnNotFound)
	}

	s, _ := w.df.Column(col)
	if !isNumericType(s.Dtype()) {
		return nil, fmt.Errorf("column %q: cannot compute std of non-numeric type", col)
	}

	nrows := w.df.Nrows()
	result := make([]float64, nrows)

	for i := 0; i < nrows; i++ {
		windowStart, windowEnd := w.getWindowBounds(i, nrows)
		values := w.extractWindowValues(s, windowStart, windowEnd)

		if len(values) < w.minPeriods {
			result[i] = math.NaN()
		} else {
			result[i] = windowStd(values)
		}
	}

	return series.New(col+"_std", result, core.DtypeFloat64), nil
}

// Min calculates the rolling minimum for a column.
func (w *Window) Min(col string) (*series.Series[any], error) {
	if !w.df.HasColumn(col) {
		return nil, fmt.Errorf("column %q: %w", col, core.ErrColumnNotFound)
	}

	s, _ := w.df.Column(col)
	nrows := w.df.Nrows()
	result := make([]any, nrows)

	for i := 0; i < nrows; i++ {
		windowStart, windowEnd := w.getWindowBounds(i, nrows)
		values := w.extractWindowValues(s, windowStart, windowEnd)

		if len(values) < w.minPeriods {
			result[i] = nil
		} else {
			result[i] = windowMin(values)
		}
	}

	return series.New(col+"_min", result, s.Dtype()), nil
}

// Max calculates the rolling maximum for a column.
func (w *Window) Max(col string) (*series.Series[any], error) {
	if !w.df.HasColumn(col) {
		return nil, fmt.Errorf("column %q: %w", col, core.ErrColumnNotFound)
	}

	s, _ := w.df.Column(col)
	nrows := w.df.Nrows()
	result := make([]any, nrows)

	for i := 0; i < nrows; i++ {
		windowStart, windowEnd := w.getWindowBounds(i, nrows)
		values := w.extractWindowValues(s, windowStart, windowEnd)

		if len(values) < w.minPeriods {
			result[i] = nil
		} else {
			result[i] = windowMax(values)
		}
	}

	return series.New(col+"_max", result, s.Dtype()), nil
}

// Helper functions

// getWindowBounds returns the start and end indices for the window at position i.
func (w *Window) getWindowBounds(i, nrows int) (int, int) {
	if w.windowType == "expanding" {
		return 0, i + 1
	}

	if w.center {
		// Centered window
		halfSize := w.size / 2
		start := i - halfSize
		end := i + halfSize + 1

		if start < 0 {
			start = 0
		}
		if end > nrows {
			end = nrows
		}

		return start, end
	}

	// Right-aligned window (default)
	start := i - w.size + 1
	if start < 0 {
		start = 0
	}

	return start, i + 1
}

// extractWindowValues extracts non-null values from a window.
func (w *Window) extractWindowValues(s *series.Series[any], start, end int) []float64 {
	values := make([]float64, 0, end-start)

	for i := start; i < end; i++ {
		if !s.IsNull(i) {
			val := s.GetUnsafe(i)
			values = append(values, toFloat64(val))
		}
	}

	return values
}

// ewmMean calculates exponentially weighted mean.
func (w *Window) ewmMean(values []float64) float64 {
	if len(values) == 0 {
		return math.NaN()
	}

	result := values[0]
	for i := 1; i < len(values); i++ {
		result = w.alpha*values[i] + (1-w.alpha)*result
	}

	return result
}

// Window aggregation functions

func windowMean(values []float64) float64 {
	if len(values) == 0 {
		return math.NaN()
	}

	var sum float64
	for _, v := range values {
		sum += v
	}

	return sum / float64(len(values))
}

func windowSum(values []float64) float64 {
	var sum float64
	for _, v := range values {
		sum += v
	}
	return sum
}

func windowStd(values []float64) float64 {
	if len(values) < 2 {
		return math.NaN()
	}

	mean := windowMean(values)
	var sumSq float64

	for _, v := range values {
		diff := v - mean
		sumSq += diff * diff
	}

	return math.Sqrt(sumSq / float64(len(values)-1))
}

func windowMin(values []float64) any {
	if len(values) == 0 {
		return nil
	}

	min := values[0]
	for _, v := range values {
		if v < min {
			min = v
		}
	}

	return min
}

func windowMax(values []float64) any {
	if len(values) == 0 {
		return nil
	}

	max := values[0]
	for _, v := range values {
		if v > max {
			max = v
		}
	}

	return max
}
