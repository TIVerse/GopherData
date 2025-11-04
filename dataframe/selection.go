package dataframe

import (
	"fmt"

	"github.com/TIVerse/GopherData/core"
	"github.com/TIVerse/GopherData/series"
)

// Row represents a single row in a DataFrame for filtering operations.
type Row struct {
	df  *DataFrame
	idx int
}

// Get returns the value in the specified column for this row.
func (r *Row) Get(col string) (any, bool) {
	s, exists := r.df.series[col]
	if !exists {
		return nil, false
	}
	return s.Get(r.idx)
}

// Select returns a new DataFrame containing only the specified columns.
// This is a view operation (zero-copy) - the underlying data is shared.
func (df *DataFrame) Select(cols ...string) *DataFrame {
	df.mu.RLock()
	defer df.mu.RUnlock()

	// Filter valid columns
	validCols := make([]string, 0, len(cols))
	newSeries := make(map[string]*series.Series[any])

	for _, col := range cols {
		if s, exists := df.series[col]; exists {
			validCols = append(validCols, col)
			newSeries[col] = s // Share the series (view)
		}
	}

	return &DataFrame{
		columns: validCols,
		series:  newSeries,
		index:   df.index,
		nrows:   df.nrows,
	}
}

// Filter returns a new DataFrame containing only rows for which the predicate returns true.
// This creates a copy of the data for filtered rows.
func (df *DataFrame) Filter(fn func(*Row) bool) *DataFrame {
	df.mu.RLock()
	defer df.mu.RUnlock()

	// Find matching row indices
	matchingIndices := make([]int, 0, df.nrows/2) // Estimate half will match

	row := &Row{df: df}
	for i := 0; i < df.nrows; i++ {
		row.idx = i
		if fn(row) {
			matchingIndices = append(matchingIndices, i)
		}
	}

	// Create new series with filtered data
	newSeries := make(map[string]*series.Series[any])
	for _, col := range df.columns {
		oldSeries := df.series[col]
		
		// Extract matching values
		filteredData := make([]any, len(matchingIndices))
		for j, i := range matchingIndices {
			val, ok := oldSeries.Get(i)
			if ok {
				filteredData[j] = val
			} else {
				filteredData[j] = nil
			}
		}

		// Create new series
		newS := series.New(col, filteredData, oldSeries.Dtype())
		
		// Mark nulls
		for j, i := range matchingIndices {
			if oldSeries.IsNull(i) {
				newS.SetNull(j)
			}
		}

		newSeries[col] = newS
	}

	return &DataFrame{
		columns: df.columns,
		series:  newSeries,
		index:   NewRangeIndex(0, len(matchingIndices), 1),
		nrows:   len(matchingIndices),
	}
}

// Loc returns rows by label-based indexing.
func (df *DataFrame) Loc(labels ...any) (*DataFrame, error) {
	df.mu.RLock()
	defer df.mu.RUnlock()

	if df.index == nil {
		return nil, fmt.Errorf("no index set: %w", core.ErrInvalidArgument)
	}

	positions, err := df.index.Loc(labels...)
	if err != nil {
		return nil, err
	}

	return df.iloc(positions), nil
}

// Iloc returns rows by integer position.
func (df *DataFrame) Iloc(positions ...int) *DataFrame {
	df.mu.RLock()
	defer df.mu.RUnlock()

	return df.iloc(positions)
}

// iloc is the internal implementation (must be called with lock held).
func (df *DataFrame) iloc(positions []int) *DataFrame {
	// Validate positions
	for _, pos := range positions {
		if pos < 0 || pos >= df.nrows {
			// Skip invalid positions for now
			continue
		}
	}

	// Create new series with selected rows
	newSeries := make(map[string]*series.Series[any])
	for _, col := range df.columns {
		oldSeries := df.series[col]

		selectedData := make([]any, len(positions))
		for j, pos := range positions {
			if pos >= 0 && pos < df.nrows {
				val, ok := oldSeries.Get(pos)
				if ok {
					selectedData[j] = val
				} else {
					selectedData[j] = nil
				}
			}
		}

		newS := series.New(col, selectedData, oldSeries.Dtype())

		// Mark nulls
		for j, pos := range positions {
			if pos >= 0 && pos < df.nrows && oldSeries.IsNull(pos) {
				newS.SetNull(j)
			}
		}

		newSeries[col] = newS
	}

	return &DataFrame{
		columns: df.columns,
		series:  newSeries,
		index:   NewRangeIndex(0, len(positions), 1),
		nrows:   len(positions),
	}
}

// Drop returns a new DataFrame with the specified columns removed.
func (df *DataFrame) Drop(cols ...string) *DataFrame {
	df.mu.RLock()
	defer df.mu.RUnlock()

	// Create a set of columns to drop
	dropSet := make(map[string]bool)
	for _, col := range cols {
		dropSet[col] = true
	}

	// Keep columns not in drop set
	newColumns := make([]string, 0, len(df.columns))
	newSeries := make(map[string]*series.Series[any])

	for _, col := range df.columns {
		if !dropSet[col] {
			newColumns = append(newColumns, col)
			newSeries[col] = df.series[col]
		}
	}

	return &DataFrame{
		columns: newColumns,
		series:  newSeries,
		index:   df.index,
		nrows:   df.nrows,
	}
}

// SliceRows returns a new DataFrame with rows from start (inclusive) to end (exclusive).
func (df *DataFrame) SliceRows(start, end int) *DataFrame {
	df.mu.RLock()
	defer df.mu.RUnlock()

	if start < 0 {
		start = 0
	}
	if end > df.nrows {
		end = df.nrows
	}
	if start >= end {
		return &DataFrame{
			columns: df.columns,
			series:  make(map[string]*series.Series[any]),
			index:   NewRangeIndex(0, 0, 1),
			nrows:   0,
		}
	}

	newSeries := make(map[string]*series.Series[any])
	for _, col := range df.columns {
		newSeries[col] = df.series[col].Slice(start, end)
	}

	newIndex := df.index
	if df.index != nil {
		newIndex = df.index.Slice(start, end)
	}

	return &DataFrame{
		columns: df.columns,
		series:  newSeries,
		index:   newIndex,
		nrows:   end - start,
	}
}
