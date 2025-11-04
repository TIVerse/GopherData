package dataframe

import (

	"github.com/TIVerse/GopherData/core"
	"github.com/TIVerse/GopherData/series"
)

// Apply applies a function to each row and adds the result as a new column.
// The function receives a Row and returns a single value.
func (df *DataFrame) Apply(fn func(*Row) any, resultCol string) *DataFrame {
	df.mu.RLock()
	defer df.mu.RUnlock()

	results := make([]any, df.nrows)

	for i := 0; i < df.nrows; i++ {
		row := &Row{
			df:  df,
			idx: i,
		}
		results[i] = fn(row)
	}

	// Create new DataFrame with result column added
	newSeries := make(map[string]*series.Series[any])
	for col, s := range df.series {
		newSeries[col] = s
	}

	// Infer dtype from first non-nil result
	dtype := core.DtypeString
	for _, val := range results {
		if val != nil {
			dtype = inferDtypeFromValue(val)
			break
		}
	}

	resultSeries := series.New(resultCol, results, dtype)
	newSeries[resultCol] = resultSeries

	// Add column name to list
	newColumns := make([]string, len(df.columns)+1)
	copy(newColumns, df.columns)
	newColumns[len(df.columns)] = resultCol

	return &DataFrame{
		columns: newColumns,
		series:  newSeries,
		index:   df.index,
		nrows:   df.nrows,
	}
}

// ApplyColumn applies a function to each value in a column.
func (df *DataFrame) ApplyColumn(col string, fn func(any) any) *DataFrame {
	df.mu.RLock()
	defer df.mu.RUnlock()

	if !df.HasColumn(col) {
		return df.Copy()
	}

	s := df.series[col]
	newData := make([]any, s.Len())

	for i := 0; i < s.Len(); i++ {
		val, ok := s.Get(i)
		if ok {
			newData[i] = fn(val)
		} else {
			newData[i] = nil
		}
	}

	// Create new series with transformed data
	newS := series.New(col, newData, s.Dtype())
	
	// Preserve null mask
	for i := 0; i < s.Len(); i++ {
		if s.IsNull(i) {
			newS.SetNull(i)
		}
	}

	// Create new DataFrame with updated series
	newSeries := make(map[string]*series.Series[any])
	for c, ser := range df.series {
		if c == col {
			newSeries[c] = newS
		} else {
			newSeries[c] = ser
		}
	}

	return &DataFrame{
		columns: df.columns,
		series:  newSeries,
		index:   df.index,
		nrows:   df.nrows,
	}
}

// ApplyElement applies a function to selected columns element-wise.
// The function receives a map of column values for the current row.
func (df *DataFrame) ApplyElement(cols []string, fn func(map[string]any) map[string]any) *DataFrame {
	df.mu.RLock()
	defer df.mu.RUnlock()

	// Validate columns
	for _, col := range cols {
		if !df.HasColumn(col) {
			return df.Copy()
		}
	}

	// Collect results for each column
	results := make(map[string][]any)
	for _, col := range cols {
		results[col] = make([]any, df.nrows)
	}

	// Apply function to each row
	for i := 0; i < df.nrows; i++ {
		// Extract values for this row
		rowVals := make(map[string]any)
		for _, col := range cols {
			s := df.series[col]
			val, ok := s.Get(i)
			if ok {
				rowVals[col] = val
			} else {
				rowVals[col] = nil
			}
		}

		// Apply function
		transformed := fn(rowVals)

		// Store results
		for _, col := range cols {
			results[col][i] = transformed[col]
		}
	}

	// Create new DataFrame with transformed columns
	newSeries := make(map[string]*series.Series[any])
	for col, s := range df.series {
		if newData, exists := results[col]; exists {
			newS := series.New(col, newData, s.Dtype())
			
			// Mark nulls
			for i := 0; i < len(newData); i++ {
				if newData[i] == nil {
					newS.SetNull(i)
				}
			}
			
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

// Map applies a function to each value in the DataFrame.
// Returns a new DataFrame with all values transformed.
func (df *DataFrame) Map(fn func(any) any) *DataFrame {
	df.mu.RLock()
	defer df.mu.RUnlock()

	newSeries := make(map[string]*series.Series[any])

	for _, col := range df.columns {
		s := df.series[col]
		newData := make([]any, s.Len())

		for i := 0; i < s.Len(); i++ {
			val, ok := s.Get(i)
			if ok {
				newData[i] = fn(val)
			} else {
				newData[i] = nil
			}
		}

		newS := series.New(col, newData, s.Dtype())
		
		// Preserve null mask
		for i := 0; i < s.Len(); i++ {
			if s.IsNull(i) {
				newS.SetNull(i)
			}
		}

		newSeries[col] = newS
	}

	return &DataFrame{
		columns: df.columns,
		series:  newSeries,
		index:   df.index,
		nrows:   df.nrows,
	}
}

// WithColumn adds or replaces a column with the given Series.
func (df *DataFrame) WithColumn(name string, s *series.Series[any]) *DataFrame {
	df.mu.RLock()
	defer df.mu.RUnlock()

	if s.Len() != df.nrows {
		return df.Copy() // Length mismatch
	}

	newSeries := make(map[string]*series.Series[any])
	for col, ser := range df.series {
		newSeries[col] = ser
	}
	newSeries[name] = s

	// Check if column is new
	newColumns := df.columns
	if !df.HasColumn(name) {
		newColumns = make([]string, len(df.columns)+1)
		copy(newColumns, df.columns)
		newColumns[len(df.columns)] = name
	}

	return &DataFrame{
		columns: newColumns,
		series:  newSeries,
		index:   df.index,
		nrows:   df.nrows,
	}
}

// Rename renames columns in the DataFrame.
func (df *DataFrame) Rename(mapping map[string]string) *DataFrame {
	df.mu.RLock()
	defer df.mu.RUnlock()

	newSeries := make(map[string]*series.Series[any])
	newColumns := make([]string, len(df.columns))

	for i, col := range df.columns {
		newName := col
		if renamed, exists := mapping[col]; exists {
			newName = renamed
		}

		newColumns[i] = newName
		newSeries[newName] = df.series[col]
	}

	return &DataFrame{
		columns: newColumns,
		series:  newSeries,
		index:   df.index,
		nrows:   df.nrows,
	}
}

// Helper function to infer dtype from a value.
func inferDtypeFromValue(val any) core.Dtype {
	switch val.(type) {
	case int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64:
		return core.DtypeInt64
	case float32, float64:
		return core.DtypeFloat64
	case bool:
		return core.DtypeBool
	case string:
		return core.DtypeString
	default:
		return core.DtypeString
	}
}
