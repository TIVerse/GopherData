package dataframe

import (
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/TIVerse/GopherData/core"
)

// Aggregation function names
const (
	AggSum    = "sum"    // Sum of values
	AggMean   = "mean"   // Arithmetic mean
	AggMedian = "median" // 50th percentile
	AggStd    = "std"    // Sample standard deviation
	AggVar    = "var"    // Sample variance
	AggMin    = "min"    // Minimum value
	AggMax    = "max"    // Maximum value
	AggCount  = "count"  // Count non-null values
	AggSize   = "size"   // Count all values (including nulls)
	AggFirst  = "first"  // First value in group
	AggLast   = "last"   // Last value in group
)

// GroupBy represents a grouped DataFrame for aggregation operations.
type GroupBy struct {
	df      *DataFrame
	keys    []string          // Group-by column names
	groups  map[string][]int  // Group key hash → row indices
	groupKeys [][]any         // Original group key values for each group
}

// GroupBy creates a GroupBy object for aggregation operations.
func (df *DataFrame) GroupBy(cols ...string) (*GroupBy, error) {
	if len(cols) == 0 {
		return nil, fmt.Errorf("at least one column required for groupby: %w", core.ErrInvalidArgument)
	}

	df.mu.RLock()
	defer df.mu.RUnlock()

	// Validate columns exist
	for _, col := range cols {
		if !df.HasColumn(col) {
			return nil, fmt.Errorf("column %q: %w", col, core.ErrColumnNotFound)
		}
	}

	// Build groups using hash map
	groups := make(map[string][]int)
	groupKeys := make(map[string][]any)

	for i := 0; i < df.nrows; i++ {
		// Extract key values for this row
		keyValues := make([]any, len(cols))
		for j, col := range cols {
			s := df.series[col]
			val, ok := s.Get(i)
			if !ok {
				keyValues[j] = nil
			} else {
				keyValues[j] = val
			}
		}

		// Hash the key
		keyHash := hashGroupKey(keyValues)

		// Add row index to group
		groups[keyHash] = append(groups[keyHash], i)
		
		// Store key values for first occurrence
		if _, exists := groupKeys[keyHash]; !exists {
			groupKeys[keyHash] = keyValues
		}
	}

	// Convert map to slice for stable ordering
	groupKeySlice := make([][]any, 0, len(groups))
	for hash := range groups {
		groupKeySlice = append(groupKeySlice, groupKeys[hash])
	}

	return &GroupBy{
		df:        df,
		keys:      cols,
		groups:    groups,
		groupKeys: groupKeySlice,
	}, nil
}

// Agg performs single aggregation per column.
// ops maps column names to aggregation function names.
// Example: {"sales": "sum", "qty": "mean"}
func (gb *GroupBy) Agg(ops map[string]string) (*DataFrame, error) {
	if len(ops) == 0 {
		return nil, fmt.Errorf("at least one aggregation required: %w", core.ErrInvalidArgument)
	}

	// Validate columns and aggregation functions
	for col, aggFunc := range ops {
		if !gb.df.HasColumn(col) {
			return nil, fmt.Errorf("column %q: %w", col, core.ErrColumnNotFound)
		}
		if !isValidAggFunc(aggFunc) {
			return nil, fmt.Errorf("unknown aggregation function %q", aggFunc)
		}
	}

	nGroups := len(gb.groups)
	
	// Build result data
	resultData := make(map[string]any)

	// Add group key columns
	for i, keyCol := range gb.keys {
		keyData := make([]any, nGroups)
		for j, keyValues := range gb.groupKeys {
			keyData[j] = keyValues[i]
		}
		resultData[keyCol] = keyData
	}

	// Add aggregated columns
	for col, aggFunc := range ops {
		aggData := make([]any, nGroups)
		
		for i, keyValues := range gb.groupKeys {
			keyHash := hashGroupKey(keyValues)
			rowIndices := gb.groups[keyHash]
			
			// Extract values for this group
			values := make([]any, 0, len(rowIndices))
			s := gb.df.series[col]
			
			for _, idx := range rowIndices {
				val, ok := s.Get(idx)
				if ok {
					values = append(values, val)
				} else if aggFunc == AggSize {
					values = append(values, nil)
				}
			}

			// Apply aggregation
			aggResult := applyAggregation(aggFunc, values, s.Dtype())
			aggData[i] = aggResult
		}

		resultData[col] = aggData
	}

	return New(resultData)
}

// AggMultiple performs multiple aggregations per column.
// ops maps column names to slices of aggregation function names.
// Example: {"sales": ["sum", "mean", "std"], "qty": ["min", "max"]}
// Result columns: [group_keys..., sales_sum, sales_mean, sales_std, qty_min, qty_max]
func (gb *GroupBy) AggMultiple(ops map[string][]string) (*DataFrame, error) {
	if len(ops) == 0 {
		return nil, fmt.Errorf("at least one aggregation required: %w", core.ErrInvalidArgument)
	}

	// Validate
	for col, aggFuncs := range ops {
		if !gb.df.HasColumn(col) {
			return nil, fmt.Errorf("column %q: %w", col, core.ErrColumnNotFound)
		}
		for _, aggFunc := range aggFuncs {
			if !isValidAggFunc(aggFunc) {
				return nil, fmt.Errorf("unknown aggregation function %q", aggFunc)
			}
		}
	}

	nGroups := len(gb.groups)
	resultData := make(map[string]any)

	// Add group key columns
	for i, keyCol := range gb.keys {
		keyData := make([]any, nGroups)
		for j, keyValues := range gb.groupKeys {
			keyData[j] = keyValues[i]
		}
		resultData[keyCol] = keyData
	}

	// Add aggregated columns
	for col, aggFuncs := range ops {
		s := gb.df.series[col]

		for _, aggFunc := range aggFuncs {
			resultCol := fmt.Sprintf("%s_%s", col, aggFunc)
			aggData := make([]any, nGroups)

			for i, keyValues := range gb.groupKeys {
				keyHash := hashGroupKey(keyValues)
				rowIndices := gb.groups[keyHash]

				// Extract values for this group
				values := make([]any, 0, len(rowIndices))
				for _, idx := range rowIndices {
					val, ok := s.Get(idx)
					if ok {
						values = append(values, val)
					} else if aggFunc == AggSize {
						values = append(values, nil)
					}
				}

				// Apply aggregation
				aggResult := applyAggregation(aggFunc, values, s.Dtype())
				aggData[i] = aggResult
			}

			resultData[resultCol] = aggData
		}
	}

	return New(resultData)
}

// Apply applies a custom function to each group and returns a DataFrame.
func (gb *GroupBy) Apply(fn func(*DataFrame) any) (*DataFrame, error) {
	resultData := make(map[string]any)

	// Add group key columns
	for i, keyCol := range gb.keys {
		keyData := make([]any, len(gb.groupKeys))
		for j, keyValues := range gb.groupKeys {
			keyData[j] = keyValues[i]
		}
		resultData[keyCol] = keyData
	}

	// Apply function to each group
	resultCol := "result"
	resultValues := make([]any, len(gb.groupKeys))

	for i, keyValues := range gb.groupKeys {
		keyHash := hashGroupKey(keyValues)
		rowIndices := gb.groups[keyHash]

		// Create sub-DataFrame for this group
		groupDf := gb.df.Iloc(rowIndices...)

		// Apply function
		resultValues[i] = fn(groupDf)
	}

	resultData[resultCol] = resultValues

	return New(resultData)
}

// Size returns the size of each group (including nulls).
func (gb *GroupBy) Size() (*DataFrame, error) {
	return gb.Agg(map[string]string{
		gb.keys[0]: AggSize,
	})
}

// Count returns the count of non-null values in each group.
func (gb *GroupBy) Count() (*DataFrame, error) {
	// Count non-nulls in all non-key columns
	ops := make(map[string]string)
	for _, col := range gb.df.columns {
		// Skip key columns
		isKey := false
		for _, key := range gb.keys {
			if col == key {
				isKey = true
				break
			}
		}
		if !isKey {
			ops[col] = AggCount
		}
	}

	return gb.Agg(ops)
}

// Helper functions

// hashGroupKey creates a hash string from group key values.
func hashGroupKey(values []any) string {
	parts := make([]string, len(values))
	for i, val := range values {
		if val == nil {
			parts[i] = "<null>"
		} else {
			parts[i] = fmt.Sprintf("%v", val)
		}
	}
	return strings.Join(parts, "|")
}

// isValidAggFunc checks if an aggregation function name is valid.
func isValidAggFunc(name string) bool {
	validFuncs := map[string]bool{
		AggSum:    true,
		AggMean:   true,
		AggMedian: true,
		AggStd:    true,
		AggVar:    true,
		AggMin:    true,
		AggMax:    true,
		AggCount:  true,
		AggSize:   true,
		AggFirst:  true,
		AggLast:   true,
	}
	return validFuncs[name]
}

// applyAggregation applies an aggregation function to a slice of values.
func applyAggregation(aggFunc string, values []any, dtype core.Dtype) any {
	if len(values) == 0 {
		return nil
	}

	switch aggFunc {
	case AggSum:
		return aggSum(values)
	case AggMean:
		return aggMean(values)
	case AggMedian:
		return aggMedian(values)
	case AggStd:
		return aggStd(values)
	case AggVar:
		return aggVar(values)
	case AggMin:
		return aggMin(values)
	case AggMax:
		return aggMax(values)
	case AggCount:
		count := 0
		for _, v := range values {
			if v != nil {
				count++
			}
		}
		return int64(count)
	case AggSize:
		return int64(len(values))
	case AggFirst:
		for _, v := range values {
			if v != nil {
				return v
			}
		}
		return nil
	case AggLast:
		for i := len(values) - 1; i >= 0; i-- {
			if values[i] != nil {
				return values[i]
			}
		}
		return nil
	default:
		return nil
	}
}

// Aggregation implementations

func aggSum(values []any) any {
	var sum float64
	count := 0
	for _, v := range values {
		if v != nil {
			sum += toFloat64(v)
			count++
		}
	}
	if count == 0 {
		return nil
	}
	return sum
}

func aggMean(values []any) any {
	var sum float64
	count := 0
	for _, v := range values {
		if v != nil {
			sum += toFloat64(v)
			count++
		}
	}
	if count == 0 {
		return nil
	}
	return sum / float64(count)
}

func aggMedian(values []any) any {
	floats := make([]float64, 0, len(values))
	for _, v := range values {
		if v != nil {
			floats = append(floats, toFloat64(v))
		}
	}
	if len(floats) == 0 {
		return nil
	}

	sort.Float64s(floats)
	n := len(floats)
	if n%2 == 0 {
		return (floats[n/2-1] + floats[n/2]) / 2
	}
	return floats[n/2]
}

func aggStd(values []any) any {
	variance := aggVar(values)
	if variance == nil {
		return nil
	}
	return math.Sqrt(variance.(float64))
}

func aggVar(values []any) any {
	// Two-pass algorithm
	mean := aggMean(values)
	if mean == nil {
		return nil
	}
	meanVal := mean.(float64)

	var sumSq float64
	count := 0
	for _, v := range values {
		if v != nil {
			diff := toFloat64(v) - meanVal
			sumSq += diff * diff
			count++
		}
	}

	if count < 2 {
		return nil
	}

	return sumSq / float64(count-1) // Bessel's correction
}

func aggMin(values []any) any {
	var min any
	found := false

	for _, v := range values {
		if v == nil {
			continue
		}
		if !found {
			min = v
			found = true
		} else {
			if compareAny(v, min) < 0 {
				min = v
			}
		}
	}

	if !found {
		return nil
	}
	return min
}

func aggMax(values []any) any {
	var max any
	found := false

	for _, v := range values {
		if v == nil {
			continue
		}
		if !found {
			max = v
			found = true
		} else {
			if compareAny(v, max) > 0 {
				max = v
			}
		}
	}

	if !found {
		return nil
	}
	return max
}
