// Package dataframe provides a two-dimensional labeled data structure with operations.
package dataframe

import (
	"fmt"
	"strings"
	"sync"

	"github.com/TIVerse/GopherData/core"
	"github.com/TIVerse/GopherData/series"
)

// DataFrame is a two-dimensional, size-mutable, tabular data structure.
// Operations return new DataFrames (copy-on-write semantics) unless noted.
type DataFrame struct {
	columns []string                       // Column names (ordered)
	series  map[string]*series.Series[any] // Column name → typed Series
	index   core.Index                     // Row labels (from core/)
	nrows   int                            // Number of rows
	mu      sync.RWMutex                   // Thread-safety for reads
}

// Columns returns a copy of the column names.
func (df *DataFrame) Columns() []string {
	df.mu.RLock()
	defer df.mu.RUnlock()

	cols := make([]string, len(df.columns))
	copy(cols, df.columns)
	return cols
}

// Nrows returns the number of rows in the DataFrame.
func (df *DataFrame) Nrows() int {
	df.mu.RLock()
	defer df.mu.RUnlock()
	return df.nrows
}

// Ncols returns the number of columns in the DataFrame.
func (df *DataFrame) Ncols() int {
	df.mu.RLock()
	defer df.mu.RUnlock()
	return len(df.columns)
}

// Shape returns (nrows, ncols).
func (df *DataFrame) Shape() (int, int) {
	df.mu.RLock()
	defer df.mu.RUnlock()
	return df.nrows, len(df.columns)
}

// HasColumn returns true if the DataFrame has a column with the given name.
func (df *DataFrame) HasColumn(name string) bool {
	df.mu.RLock()
	defer df.mu.RUnlock()
	_, exists := df.series[name]
	return exists
}

// Column returns the Series for the given column name.
// Returns an error if the column doesn't exist.
func (df *DataFrame) Column(name string) (*series.Series[any], error) {
	df.mu.RLock()
	defer df.mu.RUnlock()

	s, exists := df.series[name]
	if !exists {
		return nil, fmt.Errorf("column %q: %w", name, core.ErrColumnNotFound)
	}

	return s, nil
}

// Index returns the index of the DataFrame.
func (df *DataFrame) Index() core.Index {
	df.mu.RLock()
	defer df.mu.RUnlock()
	return df.index
}

// SetIndex sets a new index for the DataFrame.
func (df *DataFrame) SetIndex(idx core.Index) error {
	df.mu.Lock()
	defer df.mu.Unlock()

	if idx.Len() != df.nrows {
		return fmt.Errorf("index length %d does not match number of rows %d: %w",
			idx.Len(), df.nrows, core.ErrInvalidShape)
	}

	df.index = idx
	return nil
}

// String returns a string representation of the DataFrame.
func (df *DataFrame) String() string {
	df.mu.RLock()
	defer df.mu.RUnlock()

	if df.nrows == 0 {
		return fmt.Sprintf("DataFrame(shape=(0, %d), columns=%v)", len(df.columns), df.columns)
	}

	return df.head(10)
}

// Head returns a string representation of the first n rows.
func (df *DataFrame) Head(n int) string {
	df.mu.RLock()
	defer df.mu.RUnlock()
	return df.head(n)
}

// head is the internal implementation of Head (must be called with lock held).
func (df *DataFrame) head(n int) string {
	if n > df.nrows {
		n = df.nrows
	}

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("DataFrame(shape=(%d, %d))\n", df.nrows, len(df.columns)))

	if len(df.columns) == 0 {
		return sb.String()
	}

	// Header
	sb.WriteString("     ")
	for _, col := range df.columns {
		sb.WriteString(fmt.Sprintf("%-15s ", col))
	}
	sb.WriteString("\n")

	// Rows
	for i := 0; i < n; i++ {
		sb.WriteString(fmt.Sprintf("%-4d ", i))
		for _, col := range df.columns {
			s := df.series[col]
			val, ok := s.Get(i)
			if !ok {
				sb.WriteString(fmt.Sprintf("%-15s ", "<null>"))
			} else {
				sb.WriteString(fmt.Sprintf("%-15v ", val))
			}
		}
		sb.WriteString("\n")
	}

	if n < df.nrows {
		sb.WriteString(fmt.Sprintf("... (%d more rows)\n", df.nrows-n))
	}

	return sb.String()
}

// Tail returns a string representation of the last n rows.
func (df *DataFrame) Tail(n int) string {
	df.mu.RLock()
	defer df.mu.RUnlock()

	if n > df.nrows {
		n = df.nrows
	}

	start := df.nrows - n
	if start < 0 {
		start = 0
	}

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("DataFrame(shape=(%d, %d))\n", df.nrows, len(df.columns)))

	if len(df.columns) == 0 {
		return sb.String()
	}

	// Header
	sb.WriteString("     ")
	for _, col := range df.columns {
		sb.WriteString(fmt.Sprintf("%-15s ", col))
	}
	sb.WriteString("\n")

	// Rows
	for i := start; i < df.nrows; i++ {
		sb.WriteString(fmt.Sprintf("%-4d ", i))
		for _, col := range df.columns {
			s := df.series[col]
			val, ok := s.Get(i)
			if !ok {
				sb.WriteString(fmt.Sprintf("%-15s ", "<null>"))
			} else {
				sb.WriteString(fmt.Sprintf("%-15v ", val))
			}
		}
		sb.WriteString("\n")
	}

	return sb.String()
}

// Copy returns a deep copy of the DataFrame.
func (df *DataFrame) Copy() *DataFrame {
	df.mu.RLock()
	defer df.mu.RUnlock()

	newDf := &DataFrame{
		columns: make([]string, len(df.columns)),
		series:  make(map[string]*series.Series[any]),
		index:   df.index,
		nrows:   df.nrows,
	}

	copy(newDf.columns, df.columns)

	for name, s := range df.series {
		newDf.series[name] = s.Copy()
	}

	return newDf
}

// Empty returns true if the DataFrame has no rows.
func (df *DataFrame) Empty() bool {
	df.mu.RLock()
	defer df.mu.RUnlock()
	return df.nrows == 0
}
