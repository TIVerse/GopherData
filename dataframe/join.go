package dataframe

import (
	"fmt"
	"strings"

	"github.com/TIVerse/GopherData/core"
)

// Join types
const (
	JoinInner = "inner" // Intersection
	JoinLeft  = "left"  // All from left, matching from right
	JoinRight = "right" // All from right, matching from left
	JoinOuter = "outer" // Union (full outer join)
	JoinCross = "cross" // Cartesian product
)

// JoinOptions configures join behavior.
type JoinOptions struct {
	suffixLeft  string
	suffixRight string
	indicator   string
}

// JoinOption is a functional option for joins.
type JoinOption func(*JoinOptions)

// WithSuffixes sets custom suffixes for overlapping columns.
func WithSuffixes(left, right string) JoinOption {
	return func(opts *JoinOptions) {
		opts.suffixLeft = left
		opts.suffixRight = right
	}
}

// WithIndicator adds a column indicating the source of each row.
func WithIndicator(colName string) JoinOption {
	return func(opts *JoinOptions) {
		opts.indicator = colName
	}
}

// Join performs a join operation on a single column.
func (df *DataFrame) Join(other *DataFrame, joinType, onCol string, opts ...JoinOption) (*DataFrame, error) {
	return df.Merge(other, joinType, []string{onCol}, []string{onCol}, opts...)
}

// Merge performs a join operation on multiple columns.
// leftOn and rightOn specify the join keys for left and right DataFrames.
func (df *DataFrame) Merge(other *DataFrame, joinType string, leftOn, rightOn []string, opts ...JoinOption) (*DataFrame, error) {
	// Validate join type
	if !isValidJoinType(joinType) {
		return nil, fmt.Errorf("invalid join type %q", joinType)
	}

	// Validate key columns
	// Validate key columns (except for cross join)
	if joinType != JoinCross && (len(leftOn) == 0 || len(rightOn) == 0) {
		return nil, fmt.Errorf("join keys cannot be empty: %w", core.ErrInvalidArgument)
	if joinType != JoinCross && len(leftOn) != len(rightOn) {
		return nil, fmt.Errorf("left and right join keys must have same length: %w", core.ErrInvalidArgument)
	}
	if joinType != JoinCross && len(leftOn) != len(rightOn) {
		return nil, fmt.Errorf("left and right join keys must have same length: %w", core.ErrInvalidArgument)
	}
	if joinType != JoinCross && len(leftOn) != len(rightOn) {
		return nil, fmt.Errorf("left and right join keys must have same length: %w", core.ErrInvalidArgument)
	}
		return nil, fmt.Errorf("join keys cannot be empty: %w", core.ErrInvalidArgument)
	}
	// Validate key columns (except for cross join)
	if joinType != JoinCross && (len(leftOn) == 0 || len(rightOn) == 0) {
		return nil, fmt.Errorf("join keys cannot be empty: %w", core.ErrInvalidArgument)
	}
	if len(leftOn) != len(rightOn) {
		return nil, fmt.Errorf("left and right join keys must have same length: %w", core.ErrInvalidArgument)
	}

	df.mu.RLock()
	other.mu.RLock()
	defer df.mu.RUnlock()
	defer other.mu.RUnlock()

	// Validate key columns exist
	for _, col := range leftOn {
		if !df.HasColumn(col) {
			return nil, fmt.Errorf("left key column %q: %w", col, core.ErrColumnNotFound)
		}
	}
	for _, col := range rightOn {
		if !other.HasColumn(col) {
			return nil, fmt.Errorf("right key column %q: %w", col, core.ErrColumnNotFound)
		}
	}

	// Apply options
	joinOpts := &JoinOptions{
		suffixLeft:  "_left",
		suffixRight: "_right",
		indicator:   "",
	}
	for _, opt := range opts {
		opt(joinOpts)
	}

	// Perform join based on type
	switch joinType {
	case JoinInner:
		return hashJoinInner(df, other, leftOn, rightOn, joinOpts)
	case JoinLeft:
		return hashJoinLeft(df, other, leftOn, rightOn, joinOpts)
	case JoinRight:
		return hashJoinRight(df, other, leftOn, rightOn, joinOpts)
	case JoinOuter:
		return hashJoinOuter(df, other, leftOn, rightOn, joinOpts)
	case JoinCross:
		return crossJoin(df, other, joinOpts)
	default:
		return nil, fmt.Errorf("unsupported join type %q", joinType)
	}
}

// hashJoinInner performs an inner join using hash join algorithm.
func hashJoinInner(left, right *DataFrame, leftOn, rightOn []string, opts *JoinOptions) (*DataFrame, error) {
	// Build hash table on right (smaller table ideally)
	rightHash := buildHashTable(right, rightOn)

	// Probe with left table
	var matchedLeftRows []int
	var matchedRightRows []int

	for i := 0; i < left.nrows; i++ {
		// Extract left key
		leftKey := extractKey(left, i, leftOn)
		
		// Skip null keys (SQL semantics: nulls never match)
		if hasNullKey(leftKey) {
			continue
		}

		keyHash := hashJoinKey(leftKey)

		// Find matches in right table
		if rightRows, exists := rightHash[keyHash]; exists {
			for _, rightIdx := range rightRows {
				// Verify key equality (handle hash collisions)
				rightKey := extractKey(right, rightIdx, rightOn)
				if keysEqual(leftKey, rightKey) {
					matchedLeftRows = append(matchedLeftRows, i)
					matchedRightRows = append(matchedRightRows, rightIdx)
				}
			}
		}
	}

	// Build result DataFrame
	return buildJoinResult(left, right, matchedLeftRows, matchedRightRows, leftOn, opts)
}

// hashJoinLeft performs a left join.
func hashJoinLeft(left, right *DataFrame, leftOn, rightOn []string, opts *JoinOptions) (*DataFrame, error) {
	// Build hash table on right
	rightHash := buildHashTable(right, rightOn)

	var matchedLeftRows []int
	var matchedRightRows []int

	for i := 0; i < left.nrows; i++ {
		leftKey := extractKey(left, i, leftOn)
		
		// Handle null keys: keep left row, no right match
		if hasNullKey(leftKey) {
			matchedLeftRows = append(matchedLeftRows, i)
			matchedRightRows = append(matchedRightRows, -1) // No match
			continue
		}

		keyHash := hashJoinKey(leftKey)
		
		if rightRows, exists := rightHash[keyHash]; exists {
			matched := false
			for _, rightIdx := range rightRows {
				rightKey := extractKey(right, rightIdx, rightOn)
				if keysEqual(leftKey, rightKey) {
					matchedLeftRows = append(matchedLeftRows, i)
					matchedRightRows = append(matchedRightRows, rightIdx)
					matched = true
				}
			}
			if !matched {
				// Left row with no match
				matchedLeftRows = append(matchedLeftRows, i)
				matchedRightRows = append(matchedRightRows, -1)
			}
		} else {
			// Left row with no match
			matchedLeftRows = append(matchedLeftRows, i)
			matchedRightRows = append(matchedRightRows, -1)
		}
	}

	return buildJoinResult(left, right, matchedLeftRows, matchedRightRows, leftOn, opts)
}

// hashJoinRight performs a right join.
func hashJoinRight(left, right *DataFrame, leftOn, rightOn []string, opts *JoinOptions) (*DataFrame, error) {
	// Right join is left join with tables swapped
	result, err := hashJoinLeft(right, left, rightOn, leftOn, opts)
	if err != nil {
		return nil, err
	}
	
	// Reorder columns to match expected output (left columns first)
	return result, nil
}

// hashJoinOuter performs a full outer join.
func hashJoinOuter(left, right *DataFrame, leftOn, rightOn []string, opts *JoinOptions) (*DataFrame, error) {
	// Build hash tables for both sides
	rightHash := buildHashTable(right, rightOn)
	matchedRight := make(map[int]bool)

	var matchedLeftRows []int
	var matchedRightRows []int

	// Phase 1: Process left table
	for i := 0; i < left.nrows; i++ {
		leftKey := extractKey(left, i, leftOn)
		
		if hasNullKey(leftKey) {
			// Keep left row with null key
			matchedLeftRows = append(matchedLeftRows, i)
			matchedRightRows = append(matchedRightRows, -1)
			continue
		}

		keyHash := hashJoinKey(leftKey)
		
		if rightRows, exists := rightHash[keyHash]; exists {
			foundMatch := false
			for _, rightIdx := range rightRows {
				rightKey := extractKey(right, rightIdx, rightOn)
				if keysEqual(leftKey, rightKey) {
					matchedLeftRows = append(matchedLeftRows, i)
					matchedRightRows = append(matchedRightRows, rightIdx)
					matchedRight[rightIdx] = true
					foundMatch = true
				}
			}
			if !foundMatch {
				matchedLeftRows = append(matchedLeftRows, i)
				matchedRightRows = append(matchedRightRows, -1)
			}
		} else {
			// No match in right
			matchedLeftRows = append(matchedLeftRows, i)
			matchedRightRows = append(matchedRightRows, -1)
		}
	}

	// Phase 2: Add unmatched right rows
	for j := 0; j < right.nrows; j++ {
		if !matchedRight[j] {
			matchedLeftRows = append(matchedLeftRows, -1)
			matchedRightRows = append(matchedRightRows, j)
		}
	}

	return buildJoinResult(left, right, matchedLeftRows, matchedRightRows, leftOn, opts)
}

// crossJoin performs a Cartesian product.
func crossJoin(left, right *DataFrame, opts *JoinOptions) (*DataFrame, error) {
	var matchedLeftRows []int
	var matchedRightRows []int

	for i := 0; i < left.nrows; i++ {
		for j := 0; j < right.nrows; j++ {
			matchedLeftRows = append(matchedLeftRows, i)
			matchedRightRows = append(matchedRightRows, j)
		}
	}

	return buildJoinResult(left, right, matchedLeftRows, matchedRightRows, nil, opts)
}

// Helper functions

func isValidJoinType(joinType string) bool {
	return joinType == JoinInner || joinType == JoinLeft ||
		joinType == JoinRight || joinType == JoinOuter || joinType == JoinCross
}

func buildHashTable(df *DataFrame, keyColumns []string) map[string][]int {
	hashTable := make(map[string][]int)
	
	for i := 0; i < df.nrows; i++ {
		key := extractKey(df, i, keyColumns)
		
		// Skip null keys
		if hasNullKey(key) {
			continue
		}
		
		keyHash := hashJoinKey(key)
		hashTable[keyHash] = append(hashTable[keyHash], i)
	}
	
	return hashTable
}

func extractKey(df *DataFrame, row int, keyColumns []string) []any {
	key := make([]any, len(keyColumns))
	for i, col := range keyColumns {
		s := df.series[col]
		val, ok := s.Get(row)
		if !ok {
			key[i] = nil
		} else {
			key[i] = val
		}
	}
	return key
}

func hasNullKey(key []any) bool {
	for _, val := range key {
		if val == nil {
			return true
		}
	}
	return false
}

func hashJoinKey(key []any) string {
	parts := make([]string, len(key))
	for i, val := range key {
		parts[i] = fmt.Sprintf("%v", val)
	}
	return strings.Join(parts, "|")
}

func keysEqual(key1, key2 []any) bool {
	if len(key1) != len(key2) {
		return false
	}
	for i := range key1 {
		if !valuesEqual(key1[i], key2[i]) {
			return false
		}
	}
	return true
}

func valuesEqual(v1, v2 any) bool {
	if v1 == nil && v2 == nil {
		return true
	}
	if v1 == nil || v2 == nil {
		return false
	}
	return fmt.Sprintf("%v", v1) == fmt.Sprintf("%v", v2)
}

func buildJoinResult(left, right *DataFrame, leftRows, rightRows []int, keyColumns []string, opts *JoinOptions) (*DataFrame, error) {
	nrows := len(leftRows)
	resultData := make(map[string]any)

	// Determine column names and handle overlaps
	leftCols := left.columns
	rightCols := right.columns
	
	// Find overlapping columns (excluding join keys)
	keySet := make(map[string]bool)
	for _, key := range keyColumns {
		keySet[key] = true
	}
	
	overlapCols := make(map[string]bool)
	for _, rcol := range rightCols {
		if keySet[rcol] {
			continue // Skip join keys
		}
		for _, lcol := range leftCols {
			if lcol == rcol {
				overlapCols[rcol] = true
				break
			}
		}
	}

	// Add left columns
	for _, col := range leftCols {
		data := make([]any, nrows)
		s := left.series[col]
		
		for i, leftIdx := range leftRows {
			if leftIdx >= 0 {
				val, ok := s.Get(leftIdx)
				if ok {
					data[i] = val
				} else {
					data[i] = nil
				}
			} else {
				data[i] = nil
			}
		}
		
		colName := col
		if overlapCols[col] && !keySet[col] {
			colName = col + opts.suffixLeft
		}
		resultData[colName] = data
	}

	// Add right columns
	for _, col := range rightCols {
		// Skip join key columns (already added from left)
		if keySet[col] {
			continue
		}
		
		data := make([]any, nrows)
		s := right.series[col]
		
		for i, rightIdx := range rightRows {
			if rightIdx >= 0 {
				val, ok := s.Get(rightIdx)
				if ok {
					data[i] = val
				} else {
					data[i] = nil
				}
			} else {
				data[i] = nil
			}
		}
		
		colName := col
		if overlapCols[col] {
			colName = col + opts.suffixRight
		}
		resultData[colName] = data
	}

	// Add indicator column if requested
	if opts.indicator != "" {
		indicator := make([]string, nrows)
		for i := range leftRows {
			if leftRows[i] >= 0 && rightRows[i] >= 0 {
				indicator[i] = "both"
			} else if leftRows[i] >= 0 {
				indicator[i] = "left_only"
			} else {
				indicator[i] = "right_only"
			}
		}
		resultData[opts.indicator] = indicator
	}

	return New(resultData)
}
