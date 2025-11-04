package imputers

import (
	"fmt"
	"math"
	"sort"

	"github.com/TIVerse/GopherData/dataframe"
	seriesPkg "github.com/TIVerse/GopherData/series"
)

// KNNImputer fills missing values using k-Nearest Neighbors.
// For each sample with missing values, it finds k nearest neighbors
// and uses their values (weighted by distance) for imputation.
type KNNImputer struct {
	// NNeighbors is the number of neighbors to use
	NNeighbors int
	
	// Weights specifies the weight function:
	// - "uniform": all neighbors have equal weight
	// - "distance": weight inversely proportional to distance
	Weights string
	
	// Reference data for finding neighbors
	referenceData *dataframe.DataFrame
	fitted        bool
}

// NewKNNImputer creates a new KNNImputer.
func NewKNNImputer(nNeighbors int) *KNNImputer {
	return &KNNImputer{
		NNeighbors: nNeighbors,
		Weights:    "uniform",
		fitted:     false,
	}
}

// Fit stores the reference data for neighbor lookup.
func (k *KNNImputer) Fit(df *dataframe.DataFrame, _ ...string) error {
	if k.NNeighbors <= 0 {
		return fmt.Errorf("NNeighbors must be positive")
	}
	
	// Store a copy of the training data
	k.referenceData = df.Copy()
	k.fitted = true
	return nil
}

// Transform applies KNN imputation to the data.
func (k *KNNImputer) Transform(df *dataframe.DataFrame) (*dataframe.DataFrame, error) {
	if !k.fitted {
		return nil, fmt.Errorf("imputer not fitted")
	}
	
	result := df.Copy()
	cols := df.Columns()
	
	// For each column with missing values, impute using KNN
	for _, col := range cols {
		colSeries, _ := result.Column(col)
		
		// Create new data slice for this column
		newData := make([]any, colSeries.Len())
		for i := 0; i < colSeries.Len(); i++ {
			val, ok := colSeries.Get(i)
			if ok {
				newData[i] = val
			} else {
				// Value is missing - find neighbors and impute
				neighbors := k.findNeighbors(result, i, k.NNeighbors)
				imputedValue := k.computeImputedValue(col, neighbors)
				newData[i] = imputedValue
			}
		}
		
		// Replace column with imputed data
		newSeries := seriesPkg.New(col, newData, colSeries.Dtype())
		result = result.WithColumn(col, newSeries)
	}
	
	return result, nil
}

// FitTransform fits the imputer and transforms the data in one step.
func (k *KNNImputer) FitTransform(df *dataframe.DataFrame, target ...string) (*dataframe.DataFrame, error) {
	if err := k.Fit(df, target...); err != nil {
		return nil, err
	}
	return k.Transform(df)
}

// IsFitted returns true if the imputer has been fitted.
func (k *KNNImputer) IsFitted() bool {
	return k.fitted
}

// findNeighbors finds the k nearest neighbors to the given row.
// Returns slice of (row_index, distance) pairs.
func (k *KNNImputer) findNeighbors(df *dataframe.DataFrame, rowIdx int, nNeighbors int) []neighborInfo {
	neighbors := make([]neighborInfo, 0)
	
	// For each row in reference data
	for i := 0; i < k.referenceData.Nrows(); i++ {
		// Compute distance using complete features only
		distance := k.computeDistance(df, rowIdx, k.referenceData, i)
		neighbors = append(neighbors, neighborInfo{index: i, distance: distance})
	}
	
	// Sort by distance
	sort.Slice(neighbors, func(i, j int) bool {
		return neighbors[i].distance < neighbors[j].distance
	})
	
	// Take top nNeighbors
	if len(neighbors) > nNeighbors {
		neighbors = neighbors[:nNeighbors]
	}
	
	return neighbors
}

// computeDistance computes Euclidean distance between two rows using complete features.
func (k *KNNImputer) computeDistance(df1 *dataframe.DataFrame, idx1 int, df2 *dataframe.DataFrame, idx2 int) float64 {
	var sumSq float64
	count := 0
	
	cols := df1.Columns()
	for _, col := range cols {
		s1, _ := df1.Column(col)
		s2, _ := df2.Column(col)
		
		v1, ok1 := s1.Get(idx1)
		v2, ok2 := s2.Get(idx2)
		
		if !ok1 || !ok2 {
			continue // Skip if either is missing
		}
		
		diff := toFloat64Impute(v1) - toFloat64Impute(v2)
		sumSq += diff * diff
		count++
	}
	
	if count == 0 {
		return math.Inf(1)
	}
	
	return math.Sqrt(sumSq)
}

// computeImputedValue computes the imputed value from neighbors.
func (k *KNNImputer) computeImputedValue(col string, neighbors []neighborInfo) float64 {
	if len(neighbors) == 0 {
		return 0
	}
	
	var weightedSum float64
	var weightSum float64
	
	for _, neighbor := range neighbors {
		series, _ := k.referenceData.Column(col)
		val, ok := series.Get(neighbor.index)
		if !ok {
			continue
		}
		
		weight := 1.0
		if k.Weights == "distance" {
			if neighbor.distance > 0 {
				weight = 1.0 / neighbor.distance
			}
		}
		
		weightedSum += toFloat64Impute(val) * weight
		weightSum += weight
	}
	
	if weightSum == 0 {
		return 0
	}
	
	return weightedSum / weightSum
}

type neighborInfo struct {
	index    int
	distance float64
}
