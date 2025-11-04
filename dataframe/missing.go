package dataframe

import (

	"github.com/TIVerse/GopherData/series"
)

// DropNAOptions configures DropNA behavior.
type DropNAOptions struct {
	thresh  int      // Minimum number of non-null values required
	subset  []string // Only consider these columns
	howAny  bool     // Drop if any null (default: all nulls)
}

// DropNAOption is a functional option for DropNA.
type DropNAOption func(*DropNAOptions)

// Thresh sets the minimum number of non-null values required to keep a row.
func Thresh(n int) DropNAOption {
	return func(opts *DropNAOptions) {
		opts.thresh = n
	}
}

// Subset specifies columns to consider for null checking.
func Subset(cols []string) DropNAOption {
	return func(opts *DropNAOptions) {
		opts.subset = cols
	}
}

// HowAny drops rows with any null value (default behavior).
func HowAny() DropNAOption {
	return func(opts *DropNAOptions) {
		opts.howAny = true
	}
}

// HowAll drops rows only if all values are null.
func HowAll() DropNAOption {
	return func(opts *DropNAOptions) {
		opts.howAny = false
	}
}

// DropNA returns a new DataFrame with rows containing null values removed.
func (df *DataFrame) DropNA(opts ...DropNAOption) *DataFrame {
	df.mu.RLock()
	defer df.mu.RUnlock()

	// Apply options
	dropOpts := &DropNAOptions{
		thresh:  -1,
		subset:  df.columns,
		howAny:  true,
	}
	for _, opt := range opts {
		opt(dropOpts)
	}

	// Validate subset columns
	checkCols := dropOpts.subset
	if len(checkCols) == 0 {
		checkCols = df.columns
	}
	for _, col := range checkCols {
		if !df.HasColumn(col) {
			checkCols = df.columns
			break
		}
	}

	// Find rows to keep
	keepRows := make([]int, 0, df.nrows)

	for i := 0; i < df.nrows; i++ {
		nonNullCount := 0
		nullCount := 0

		for _, col := range checkCols {
			s := df.series[col]
			if s.IsNull(i) {
				nullCount++
			} else {
				nonNullCount++
			}
		}

		// Decide whether to keep row
		keep := false

		if dropOpts.thresh >= 0 {
			// Threshold mode: keep if >= thresh non-nulls
			keep = nonNullCount >= dropOpts.thresh
		} else if dropOpts.howAny {
			// Any mode: keep if no nulls
			keep = nullCount == 0
		} else {
			// All mode: keep if not all nulls
			keep = nonNullCount > 0
		}

		if keep {
			keepRows = append(keepRows, i)
		}
	}

	// Return filtered DataFrame
	return df.iloc(keepRows)
}

// FillNA returns a new DataFrame with null values replaced by the given value.
func (df *DataFrame) FillNA(value any) *DataFrame {
	df.mu.RLock()
	defer df.mu.RUnlock()

	newSeries := make(map[string]*series.Series[any])

	for _, col := range df.columns {
		s := df.series[col]
		newS := s.FillNA(value)
		newSeries[col] = newS
	}

	return &DataFrame{
		columns: df.columns,
		series:  newSeries,
		index:   df.index,
		nrows:   df.nrows,
	}
}

// FillNAColumn returns a new DataFrame with nulls in a specific column replaced.
func (df *DataFrame) FillNAColumn(col string, value any) *DataFrame {
	df.mu.RLock()
	defer df.mu.RUnlock()

	if !df.HasColumn(col) {
		return df.Copy()
	}

	newSeries := make(map[string]*series.Series[any])

	for _, c := range df.columns {
		if c == col {
			s := df.series[c]
			newS := s.FillNA(value)
			newSeries[c] = newS
		} else {
			newSeries[c] = df.series[c]
		}
	}

	return &DataFrame{
		columns: df.columns,
		series:  newSeries,
		index:   df.index,
		nrows:   df.nrows,
	}
}

// FillNADict returns a new DataFrame with nulls replaced using a column-specific map.
func (df *DataFrame) FillNADict(values map[string]any) *DataFrame {
	df.mu.RLock()
	defer df.mu.RUnlock()

	newSeries := make(map[string]*series.Series[any])

	for _, col := range df.columns {
		s := df.series[col]
		
		if fillValue, exists := values[col]; exists {
			newS := s.FillNA(fillValue)
			newSeries[col] = newS
		} else {
			newSeries[col] = s
		}
	}

	return &DataFrame{
		columns: df.columns,
		series:  newSeries,
		index:   df.index,
		nrows:   df.nrows,
	}
}

// InterpolateOptions configures interpolation behavior.
type InterpolateOptions struct {
	limit int // Maximum number of consecutive nulls to fill
}

// InterpolateOption is a functional option for Interpolate.
type InterpolateOption func(*InterpolateOptions)

// Limit sets the maximum number of consecutive nulls to fill.
func Limit(n int) InterpolateOption {
	return func(opts *InterpolateOptions) {
		opts.limit = n
	}
}

// Interpolate fills null values using interpolation.
// method can be "linear", "ffill", or "bfill".
func (df *DataFrame) Interpolate(method string, opts ...InterpolateOption) *DataFrame {
	df.mu.RLock()
	defer df.mu.RUnlock()

	// Apply options
	interpOpts := &InterpolateOptions{
		limit: -1, // No limit by default
	}
	for _, opt := range opts {
		opt(interpOpts)
	}

	newSeries := make(map[string]*series.Series[any])

	for _, col := range df.columns {
		s := df.series[col]
		
		// Only interpolate numeric columns
		if !isNumericType(s.Dtype()) {
			newSeries[col] = s
			continue
		}

		newS := interpolateSeries(s, method, interpOpts.limit)
		newSeries[col] = newS
	}

	return &DataFrame{
		columns: df.columns,
		series:  newSeries,
		index:   df.index,
		nrows:   df.nrows,
	}
}

// interpolateSeries performs interpolation on a single series.
func interpolateSeries(s *series.Series[any], method string, limit int) *series.Series[any] {
	data := s.Data()
	newData := make([]any, len(data))
	copy(newData, data)

	switch method {
	case "linear":
		interpolateLinear(newData, s, limit)
	case "ffill":
		interpolateFFill(newData, s, limit)
	case "bfill":
		interpolateBFill(newData, s, limit)
	default:
		// Unknown method, return copy
		return s.Copy()
	}

	// Create new series
	newS := series.New(s.Name(), newData, s.Dtype())
	
	// Mark remaining nulls
	for i := 0; i < len(newData); i++ {
		if s.IsNull(i) && newData[i] == nil {
			newS.SetNull(i)
		}
	}

	return newS
}

// interpolateLinear performs linear interpolation.
func interpolateLinear(data []any, s *series.Series[any], limit int) {
	n := len(data)

	for i := 0; i < n; i++ {
		if !s.IsNull(i) {
			continue
		}

		// Find previous non-null value
		var prevIdx int = -1
		var prevVal float64
		for j := i - 1; j >= 0; j-- {
			if !s.IsNull(j) {
				prevIdx = j
				prevVal = toFloat64(s.GetUnsafe(j))
				break
			}
		}

		// Find next non-null value
		var nextIdx int = -1
		var nextVal float64
		for j := i + 1; j < n; j++ {
			if !s.IsNull(j) {
				nextIdx = j
				nextVal = toFloat64(s.GetUnsafe(j))
				break
			}
		}

		// Interpolate if we have both boundaries
		if prevIdx >= 0 && nextIdx >= 0 {
			gap := nextIdx - prevIdx
			if limit >= 0 && gap-1 > limit {
				continue // Gap too large
			}

			// Linear interpolation
			fraction := float64(i-prevIdx) / float64(gap)
			interpolated := prevVal + fraction*(nextVal-prevVal)
			data[i] = interpolated
		}
	}
}

// interpolateFFill performs forward fill.
func interpolateFFill(data []any, s *series.Series[any], limit int) {
	var lastValue any
	consecutiveNulls := 0

	for i := 0; i < len(data); i++ {
		if !s.IsNull(i) {
			lastValue = s.GetUnsafe(i)
			consecutiveNulls = 0
		} else if lastValue != nil {
			if limit < 0 || consecutiveNulls < limit {
				data[i] = lastValue
				consecutiveNulls++
			}
		}
	}
}

// interpolateBFill performs backward fill.
func interpolateBFill(data []any, s *series.Series[any], limit int) {
	var nextValue any
	consecutiveNulls := 0

	for i := len(data) - 1; i >= 0; i-- {
		if !s.IsNull(i) {
			nextValue = s.GetUnsafe(i)
			consecutiveNulls = 0
		} else if nextValue != nil {
			if limit < 0 || consecutiveNulls < limit {
				data[i] = nextValue
				consecutiveNulls++
			}
		}
	}
}

// IsNA returns a DataFrame of boolean values indicating null positions.
func (df *DataFrame) IsNA() (*DataFrame, error) {
	df.mu.RLock()
	defer df.mu.RUnlock()

	resultData := make(map[string]any)

	for _, col := range df.columns {
		s := df.series[col]
		nulls := make([]bool, df.nrows)
		
		for i := 0; i < df.nrows; i++ {
			nulls[i] = s.IsNull(i)
		}
		
		resultData[col] = nulls
	}

	return New(resultData)
}

// NotNA returns a DataFrame of boolean values indicating non-null positions.
func (df *DataFrame) NotNA() (*DataFrame, error) {
	df.mu.RLock()
	defer df.mu.RUnlock()

	resultData := make(map[string]any)

	for _, col := range df.columns {
		s := df.series[col]
		notNulls := make([]bool, df.nrows)
		
		for i := 0; i < df.nrows; i++ {
			notNulls[i] = !s.IsNull(i)
		}
		
		resultData[col] = notNulls
	}

	return New(resultData)
}
