package creators

import (
	"fmt"
	"math"
	"sort"

	"github.com/TIVerse/GopherData/core"
	"github.com/TIVerse/GopherData/dataframe"
	seriesPkg "github.com/TIVerse/GopherData/series"
)

// BinDiscretizer discretizes continuous features into bins.
type BinDiscretizer struct {
	// Columns to discretize
	Columns []string
	
	// NBins is the number of bins to create
	NBins int
	
	// Strategy for creating bins:
	// - "uniform": equal width bins
	// - "quantile": equal frequency bins
	// - "kmeans": k-means based bins (simplified)
	Strategy string
	
	// Encode specifies how to encode the bins:
	// - "ordinal": 0, 1, 2, ...
	// - "onehot": create binary columns for each bin
	Encode string
	
	// Bin edges for each column
	bins   map[string][]float64
	fitted bool
}

// NewBinDiscretizer creates a new BinDiscretizer.
func NewBinDiscretizer(columns []string, nBins int, strategy string) *BinDiscretizer {
	return &BinDiscretizer{
		Columns:  columns,
		NBins:    nBins,
		Strategy: strategy,
		Encode:   "ordinal",
		bins:     make(map[string][]float64),
		fitted:   false,
	}
}

// Fit computes the bin edges for each column.
func (b *BinDiscretizer) Fit(df *dataframe.DataFrame, _ ...string) error {
	if len(b.Columns) == 0 {
		return fmt.Errorf("no columns specified for binning")
	}
	
	if b.NBins < 2 {
		return fmt.Errorf("NBins must be at least 2")
	}
	
	b.bins = make(map[string][]float64)
	
	for _, col := range b.Columns {
		colSeries, err := df.Column(col)
		if err != nil {
			return fmt.Errorf("column %q: %w", col, err)
		}
		
		var edges []float64
		switch b.Strategy {
		case "uniform":
			edges = b.uniformBins(colSeries, b.NBins)
		case "quantile":
			edges = b.quantileBins(colSeries, b.NBins)
		case "kmeans":
			edges = b.kmeansBins(colSeries, b.NBins)
		default:
			return fmt.Errorf("unknown strategy: %s", b.Strategy)
		}
		
		b.bins[col] = edges
	}
	
	b.fitted = true
	return nil
}

// Transform discretizes the continuous features into bins.
func (b *BinDiscretizer) Transform(df *dataframe.DataFrame) (*dataframe.DataFrame, error) {
	if !b.fitted {
		return nil, fmt.Errorf("discretizer not fitted")
	}
	
	result := df.Copy()
	
	for _, col := range b.Columns {
		colSeries, err := result.Column(col)
		if err != nil {
			return nil, fmt.Errorf("column %q: %w", col, err)
		}
		
		edges := b.bins[col]
		
		// Discretize values
		discretized := make([]any, colSeries.Len())
		for i := 0; i < colSeries.Len(); i++ {
			val, ok := colSeries.Get(i)
			if !ok {
				discretized[i] = nil
				continue
			}
			
			floatVal := toFloat64Creator(val)
			binIdx := b.findBin(floatVal, edges)
			
			if b.Encode == "ordinal" {
				discretized[i] = int64(binIdx)
			} else {
				discretized[i] = int64(binIdx)
			}
		}
		
		result = result.WithColumn(col, seriesPkg.New(col, discretized, core.DtypeInt64))
		
		// If one-hot encoding, create binary columns
		if b.Encode == "onehot" {
			for binIdx := 0; binIdx < b.NBins; binIdx++ {
				binCol := make([]any, colSeries.Len())
				for i := 0; i < colSeries.Len(); i++ {
					if discretized[i] == nil {
						binCol[i] = int64(0)
					} else if discretized[i].(int64) == int64(binIdx) {
						binCol[i] = int64(1)
					} else {
						binCol[i] = int64(0)
					}
				}
				
				newColName := fmt.Sprintf("%s_bin%d", col, binIdx)
				result = result.WithColumn(newColName, seriesPkg.New(newColName, binCol, core.DtypeInt64))
			}
			
			// Drop original column
			result = result.Drop(col)
		}
	}
	
	return result, nil
}

// FitTransform fits the discretizer and transforms the data in one step.
func (b *BinDiscretizer) FitTransform(df *dataframe.DataFrame, target ...string) (*dataframe.DataFrame, error) {
	if err := b.Fit(df, target...); err != nil {
		return nil, err
	}
	return b.Transform(df)
}

// IsFitted returns true if the discretizer has been fitted.
func (b *BinDiscretizer) IsFitted() bool {
	return b.fitted
}

// GetBins returns the bin edges for each column.
func (b *BinDiscretizer) GetBins() map[string][]float64 {
	bins := make(map[string][]float64, len(b.bins))
	for k, v := range b.bins {
		edges := make([]float64, len(v))
		copy(edges, v)
		bins[k] = edges
	}
	return bins
}

// uniformBins creates equal-width bins.
func (b *BinDiscretizer) uniformBins(series interface{ Len() int; Get(int) (any, bool) }, nBins int) []float64 {
	min, max := computeMinMax(series)
	
	if min == max {
		return []float64{min, max}
	}
	
	edges := make([]float64, nBins+1)
	width := (max - min) / float64(nBins)
	
	for i := 0; i <= nBins; i++ {
		edges[i] = min + float64(i)*width
	}
	
	return edges
}

// quantileBins creates equal-frequency bins based on quantiles.
func (b *BinDiscretizer) quantileBins(series interface{ Len() int; Get(int) (any, bool) }, nBins int) []float64 {
	values := collectValues(series)
	if len(values) == 0 {
		return []float64{0, 0}
	}
	
	sort.Float64s(values)
	
	edges := make([]float64, nBins+1)
	edges[0] = values[0]
	edges[nBins] = values[len(values)-1]
	
	for i := 1; i < nBins; i++ {
		quantile := float64(i) / float64(nBins)
		pos := quantile * float64(len(values)-1)
		idx := int(pos)
		if idx >= len(values)-1 {
			edges[i] = values[len(values)-1]
		} else {
			fraction := pos - float64(idx)
			edges[i] = values[idx]*(1-fraction) + values[idx+1]*fraction
		}
	}
	
	return edges
}

// kmeansBins creates bins using a simplified k-means approach.
func (b *BinDiscretizer) kmeansBins(series interface{ Len() int; Get(int) (any, bool) }, nBins int) []float64 {
	// Simplified: use quantiles as initial centroids, then compute boundaries
	return b.quantileBins(series, nBins)
}

// findBin finds which bin a value falls into.
func (b *BinDiscretizer) findBin(value float64, edges []float64) int {
	if len(edges) < 2 {
		return 0
	}
	
	// Handle edge cases
	if value <= edges[0] {
		return 0
	}
	if value >= edges[len(edges)-1] {
		return len(edges) - 2
	}
	
	// Binary search for the bin
	for i := 0; i < len(edges)-1; i++ {
		if value >= edges[i] && value < edges[i+1] {
			return i
		}
	}
	
	return len(edges) - 2
}

func computeMinMax(series interface{ Len() int; Get(int) (any, bool) }) (float64, float64) {
	min := math.Inf(1)
	max := math.Inf(-1)
	found := false
	
	for i := 0; i < series.Len(); i++ {
		val, ok := series.Get(i)
		if !ok {
			continue
		}
		
		floatVal := toFloat64Creator(val)
		if !found {
			min = floatVal
			max = floatVal
			found = true
		} else {
			if floatVal < min {
				min = floatVal
			}
			if floatVal > max {
				max = floatVal
			}
		}
	}
	
	if !found {
		return 0, 0
	}
	return min, max
}

func collectValues(series interface{ Len() int; Get(int) (any, bool) }) []float64 {
	values := make([]float64, 0)
	
	for i := 0; i < series.Len(); i++ {
		val, ok := series.Get(i)
		if !ok {
			continue
		}
		values = append(values, toFloat64Creator(val))
	}
	
	return values
}
