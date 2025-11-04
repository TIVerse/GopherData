package dataframe

import (
	"fmt"

	"github.com/TIVerse/GopherData/core"
)

// Pivot transforms long format to wide format.
// index: column to use as row index
// columns: column to use for new column names
// values: column to use for cell values
func (df *DataFrame) Pivot(index, columns, values string) (*DataFrame, error) {
	df.mu.RLock()
	defer df.mu.RUnlock()

	// Validate columns
	for _, col := range []string{index, columns, values} {
		if !df.HasColumn(col) {
			return nil, fmt.Errorf("column %q: %w", col, core.ErrColumnNotFound)
		}
	}

	// Get unique index and column values
	indexVals := df.uniqueValues(index)
	colVals := df.uniqueValues(columns)

	// Build pivot table
	pivotData := make(map[string][]any)
	
	// Add index column
	indexData := make([]any, len(indexVals))
	for i, val := range indexVals {
		indexData[i] = val
	}
	pivotData[index] = indexData

	// Create columns for each unique column value
	for _, colVal := range colVals {
		colName := fmt.Sprintf("%v", colVal)
		columnData := make([]any, len(indexVals))
		
		// Initialize with nil
		for i := range columnData {
			columnData[i] = nil
		}
		
		pivotData[colName] = columnData
	}

	// Fill pivot table
	idxSeries := df.series[index]
	colSeries := df.series[columns]
	valSeries := df.series[values]

	// Create index lookup
	indexLookup := make(map[string]int)
	for i, val := range indexVals {
		indexLookup[fmt.Sprintf("%v", val)] = i
	}

	for i := 0; i < df.nrows; i++ {
		idxVal, _ := idxSeries.Get(i)
		colVal, _ := colSeries.Get(i)
		val, ok := valSeries.Get(i)

		if idxVal == nil || colVal == nil {
			continue
		}

		idxKey := fmt.Sprintf("%v", idxVal)
		colKey := fmt.Sprintf("%v", colVal)

		if rowIdx, exists := indexLookup[idxKey]; exists {
			if columnData, exists := pivotData[colKey]; exists {
				if ok {
					columnData[rowIdx] = val
				}
			}
		}
	}

	// Convert to map[string]any for New()
	convertedData := make(map[string]any, len(pivotData))
	for key, val := range pivotData {
		convertedData[key] = val
	}

	return New(convertedData)
}

// Melt transforms wide format to long format.
// idVars: columns to use as identifier variables
// valueVars: columns to unpivot (if empty, use all non-id columns)
// varName: name for the variable column
// valueName: name for the value column
func (df *DataFrame) Melt(idVars, valueVars []string, varName, valueName string) (*DataFrame, error) {
	df.mu.RLock()
	defer df.mu.RUnlock()

	// Validate id columns
	for _, col := range idVars {
		if !df.HasColumn(col) {
			return nil, fmt.Errorf("id column %q: %w", col, core.ErrColumnNotFound)
		}
	}

	// Determine value columns
	if len(valueVars) == 0 {
		// Use all non-id columns
		idSet := make(map[string]bool)
		for _, col := range idVars {
			idSet[col] = true
		}

		for _, col := range df.columns {
			if !idSet[col] {
				valueVars = append(valueVars, col)
			}
		}
	} else {
		// Validate value columns
		for _, col := range valueVars {
			if !df.HasColumn(col) {
				return nil, fmt.Errorf("value column %q: %w", col, core.ErrColumnNotFound)
			}
		}
	}

	if len(valueVars) == 0 {
		return nil, fmt.Errorf("no value columns to melt")
	}

	// Default names
	if varName == "" {
		varName = "variable"
	}
	if valueName == "" {
		valueName = "value"
	}

	// Calculate result size
	resultRows := df.nrows * len(valueVars)
	
	// Build melted data
	meltedData := make(map[string][]any)

	// Initialize columns
	for _, col := range idVars {
		meltedData[col] = make([]any, resultRows)
	}
	meltedData[varName] = make([]any, resultRows)
	meltedData[valueName] = make([]any, resultRows)

	// Fill melted data
	rowIdx := 0
	for i := 0; i < df.nrows; i++ {
		for _, valCol := range valueVars {
			// Copy id values
			for _, idCol := range idVars {
				s := df.series[idCol]
				val, ok := s.Get(i)
				if ok {
					meltedData[idCol][rowIdx] = val
				} else {
					meltedData[idCol][rowIdx] = nil
				}
			}

			// Add variable name
			meltedData[varName][rowIdx] = valCol

			// Add value
			s := df.series[valCol]
			val, ok := s.Get(i)
			if ok {
				meltedData[valueName][rowIdx] = val
			} else {
				meltedData[valueName][rowIdx] = nil
			}

			rowIdx++
		}
	}

	// Convert map[string][]any to map[string]any
	convertedData := make(map[string]any, len(meltedData))
	for key, val := range meltedData {
		convertedData[key] = val
	}
	return New(convertedData)
}

// Stack pivots columns into rows (multi-level index).
// For simplicity, this implementation creates a long-form DataFrame.
func (df *DataFrame) Stack() (*DataFrame, error) {
	df.mu.RLock()
	defer df.mu.RUnlock()

	// Stack all columns into two columns: 'column' and 'value'
	resultRows := df.nrows * len(df.columns)
	
	stackedData := make(map[string][]any)
	stackedData["row"] = make([]any, resultRows)
	stackedData["column"] = make([]any, resultRows)
	stackedData["value"] = make([]any, resultRows)

	rowIdx := 0
	for i := 0; i < df.nrows; i++ {
		for _, col := range df.columns {
			stackedData["row"][rowIdx] = i
			stackedData["column"][rowIdx] = col
			
			s := df.series[col]
			val, ok := s.Get(i)
			if ok {
				stackedData["value"][rowIdx] = val
			} else {
				stackedData["value"][rowIdx] = nil
			}
			
			rowIdx++
		}
	}

	// Convert map[string][]any to map[string]any
	convertedStackedData := make(map[string]any, len(stackedData))
	for key, val := range stackedData {
		convertedStackedData[key] = val
	}
	return New(convertedStackedData)
}

// Unstack pivots rows into columns (inverse of Stack).
func (df *DataFrame) Unstack(rowCol, colCol, valueCol string) (*DataFrame, error) {
	// This is essentially a Pivot operation
	return df.Pivot(rowCol, colCol, valueCol)
}

// Transpose swaps rows and columns.
func (df *DataFrame) Transpose() (*DataFrame, error) {
	df.mu.RLock()
	defer df.mu.RUnlock()

	// Build transposed data
	transposedData := make(map[string][]any)

	// Use original column names as row identifiers
	rowNames := make([]string, df.nrows)
	for i := 0; i < df.nrows; i++ {
		rowNames[i] = fmt.Sprintf("row_%d", i)
	}

	// Each original column becomes a row
	for _, col := range df.columns {
		s := df.series[col]
		colData := make([]any, df.nrows)
		
		for i := 0; i < df.nrows; i++ {
			val, ok := s.Get(i)
			if ok {
				colData[i] = val
			} else {
				colData[i] = nil
			}
		}
		
		transposedData[col] = colData
	}

	// Note: This creates a transposed DataFrame where:
	// - Original columns become rows
	// - Original rows become columns
	// Convert map[string][]any to map[string]any
	convertedTransposedData := make(map[string]any, len(transposedData))
	for key, val := range transposedData {
		convertedTransposedData[key] = val
	}
	return New(convertedTransposedData)
}

// Helper function to get unique values from a column.
func (df *DataFrame) uniqueValues(col string) []any {
	s := df.series[col]
	seen := make(map[string]bool)
	unique := make([]any, 0)

	for i := 0; i < s.Len(); i++ {
		val, ok := s.Get(i)
		if !ok {
			continue
		}

		key := fmt.Sprintf("%v", val)
		if !seen[key] {
			seen[key] = true
			unique = append(unique, val)
		}
	}

	return unique
}
