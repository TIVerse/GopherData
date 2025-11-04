package dataframe

import (
	"sort"

	"github.com/TIVerse/GopherData/core"
	"github.com/TIVerse/GopherData/series"
)

// SortOptions configures sort behavior.
type SortOptions struct {
	nullsFirst bool
	stable     bool
}

// SortOption is a functional option for sorting.
type SortOption func(*SortOptions)

// NullsFirst places null values at the beginning.
func NullsFirst() SortOption {
	return func(opts *SortOptions) {
		opts.nullsFirst = true
	}
}

// NullsLast places null values at the end (default).
func NullsLast() SortOption {
	return func(opts *SortOptions) {
		opts.nullsFirst = false
	}
}

// Stable uses stable sort algorithm.
func Stable() SortOption {
	return func(opts *SortOptions) {
		opts.stable = true
	}
}

// Sort sorts the DataFrame by a single column.
// Returns a new DataFrame with rows reordered.
func (df *DataFrame) Sort(col string, order core.Order, opts ...SortOption) *DataFrame {
	return df.SortMulti([]string{col}, []core.Order{order}, opts...)
}

// SortMulti sorts the DataFrame by multiple columns.
// Columns are sorted in order of priority (first column is primary sort key).
func (df *DataFrame) SortMulti(cols []string, orders []core.Order, opts ...SortOption) *DataFrame {
	if len(cols) == 0 || len(cols) != len(orders) {
		return df.Copy() // Return copy on invalid input
	}

	df.mu.RLock()
	defer df.mu.RUnlock()

	// Validate columns
	for _, col := range cols {
		if !df.HasColumn(col) {
			return df.Copy()
		}
	}

	// Apply options
	sortOpts := &SortOptions{
		nullsFirst: false,
		stable:     true, // Default to stable
	}
	for _, opt := range opts {
		opt(sortOpts)
	}

	// Create index array for sorting
	indices := make([]int, df.nrows)
	for i := range indices {
		indices[i] = i
	}

	// Sort indices
	sorter := &multiColumnSorter{
		df:         df,
		indices:    indices,
		cols:       cols,
		orders:     orders,
		nullsFirst: sortOpts.nullsFirst,
	}

	if sortOpts.stable {
		sort.Stable(sorter)
	} else {
		sort.Sort(sorter)
	}

	// Build sorted DataFrame
	return df.reorderRows(indices)
}

// SortIndex sorts the DataFrame by its index.
func (df *DataFrame) SortIndex(order core.Order, opts ...SortOption) *DataFrame {
	df.mu.RLock()
	defer df.mu.RUnlock()

	if df.index == nil {
		return df.Copy()
	}

	// Apply options
	sortOpts := &SortOptions{
		nullsFirst: false,
		stable:     true,
	}
	for _, opt := range opts {
		opt(sortOpts)
	}

	// Create index array
	indices := make([]int, df.nrows)
	for i := range indices {
		indices[i] = i
	}

	// Sort by index values
	sorter := &indexSorter{
		df:         df,
		indices:    indices,
		order:      order,
		nullsFirst: sortOpts.nullsFirst,
	}

	if sortOpts.stable {
		sort.Stable(sorter)
	} else {
		sort.Sort(sorter)
	}

	return df.reorderRows(indices)
}

// reorderRows creates a new DataFrame with rows in the specified order.
func (df *DataFrame) reorderRows(indices []int) *DataFrame {
	newSeries := make(map[string]*series.Series[any])

	for _, col := range df.columns {
		s := df.series[col]
		
		// Extract values in new order
		newData := make([]any, len(indices))
		for i, idx := range indices {
			val, ok := s.Get(idx)
			if ok {
				newData[i] = val
			} else {
				newData[i] = nil
			}
		}

		// Create new series
		newS := series.New(col, newData, s.Dtype())
		
		// Copy null mask
		for i, idx := range indices {
			if s.IsNull(idx) {
				newS.SetNull(i)
			}
		}

		newSeries[col] = newS
	}

	return &DataFrame{
		columns: df.columns,
		series:  newSeries,
		index:   NewRangeIndex(0, len(indices), 1),
		nrows:   len(indices),
	}
}

// multiColumnSorter implements sort.Interface for multi-column sorting.
type multiColumnSorter struct {
	df         *DataFrame
	indices    []int
	cols       []string
	orders     []core.Order
	nullsFirst bool
}

func (s *multiColumnSorter) Len() int {
	return len(s.indices)
}

func (s *multiColumnSorter) Swap(i, j int) {
	s.indices[i], s.indices[j] = s.indices[j], s.indices[i]
}

func (s *multiColumnSorter) Less(i, j int) bool {
	idx1 := s.indices[i]
	idx2 := s.indices[j]

	// Compare by each column in order
	for colIdx, col := range s.cols {
		series := s.df.series[col]
		
		val1, ok1 := series.Get(idx1)
		val2, ok2 := series.Get(idx2)

		// Handle nulls
		if !ok1 && !ok2 {
			continue // Both null, move to next column
		}
		if !ok1 {
			return s.nullsFirst
		}
		if !ok2 {
			return !s.nullsFirst
		}

		// Compare values
		cmp := compareAny(val1, val2)
		if cmp == 0 {
			continue // Equal, move to next column
		}

		// Apply sort order
		if s.orders[colIdx] == core.Ascending {
			return cmp < 0
		}
		return cmp > 0
	}

	return false // All columns equal
}

// indexSorter implements sort.Interface for index sorting.
type indexSorter struct {
	df         *DataFrame
	indices    []int
	order      core.Order
	nullsFirst bool
}

func (s *indexSorter) Len() int {
	return len(s.indices)
}

func (s *indexSorter) Swap(i, j int) {
	s.indices[i], s.indices[j] = s.indices[j], s.indices[i]
}

func (s *indexSorter) Less(i, j int) bool {
	idx1 := s.indices[i]
	idx2 := s.indices[j]

	val1 := s.df.index.Get(idx1)
	val2 := s.df.index.Get(idx2)

	// Handle nulls
	if val1 == nil && val2 == nil {
		return false
	}
	if val1 == nil {
		return s.nullsFirst
	}
	if val2 == nil {
		return !s.nullsFirst
	}

	// Compare values
	cmp := compareAny(val1, val2)
	
	if s.order == core.Ascending {
		return cmp < 0
	}
	return cmp > 0
}

// Argsort returns the indices that would sort the DataFrame.
func (df *DataFrame) Argsort(col string, order core.Order, opts ...SortOption) []int {
	df.mu.RLock()
	defer df.mu.RUnlock()

	if !df.HasColumn(col) {
		return nil
	}

	sortOpts := &SortOptions{
		nullsFirst: false,
		stable:     true,
	}
	for _, opt := range opts {
		opt(sortOpts)
	}

	indices := make([]int, df.nrows)
	for i := range indices {
		indices[i] = i
	}

	sorter := &multiColumnSorter{
		df:         df,
		indices:    indices,
		cols:       []string{col},
		orders:     []core.Order{order},
		nullsFirst: sortOpts.nullsFirst,
	}

	if sortOpts.stable {
		sort.Stable(sorter)
	} else {
		sort.Sort(sorter)
	}

	return indices
}
