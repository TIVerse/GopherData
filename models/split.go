package models

import (
	"fmt"
	"math/rand"

	"github.com/TIVerse/GopherData/dataframe"
	seriesPkg "github.com/TIVerse/GopherData/series"
)

// TrainTestSplit contains the split datasets.
type TrainTestSplit struct {
	XTrain *dataframe.DataFrame
	XTest  *dataframe.DataFrame
	YTrain *seriesPkg.Series[any]
	YTest  *seriesPkg.Series[any]
}

// TrainTestSplitFunc splits data into training and test sets.
// X: feature DataFrame
// y: target Series
// testSize: fraction of data to use for testing (0-1)
// shuffle: whether to shuffle data before splitting
// stratify: column name to stratify by (maintains class proportions)
// seed: random seed (optional, defaults to time-based)
func TrainTestSplitFunc(X *dataframe.DataFrame, y *seriesPkg.Series[any], testSize float64, shuffle bool, stratify string, seed ...int64) (TrainTestSplit, error) {
	if X.Nrows() != y.Len() {
		return TrainTestSplit{}, fmt.Errorf("x and y must have same length")
	}
	
	if testSize <= 0 || testSize >= 1 {
		return TrainTestSplit{}, fmt.Errorf("testSize must be between 0 and 1")
	}
	
	n := X.Nrows()
	testN := int(float64(n) * testSize)
	trainN := n - testN
	
	// Initialize random generator
	var rng *rand.Rand
	if len(seed) > 0 {
		rng = rand.New(rand.NewSource(seed[0]))
	} else {
		rng = rand.New(rand.NewSource(0)) // Use fixed seed for reproducibility
	}
	
	// Generate indices
	indices := make([]int, n)
	for i := range indices {
		indices[i] = i
	}
	
	// Shuffle if requested
	if shuffle {
		rng.Shuffle(len(indices), func(i, j int) {
			indices[i], indices[j] = indices[j], indices[i]
		})
	}
	
	// Handle stratification
	if stratify != "" {
		return stratifiedSplit(X, y, testSize, rng, stratify)
	}
	
	// Simple split
	trainIndices := indices[:trainN]
	testIndices := indices[trainN:]
	
	// Create splits
	XTrain, err := selectRows(X, trainIndices)
	if err != nil {
		return TrainTestSplit{}, err
	}
	
	XTest, err := selectRows(X, testIndices)
	if err != nil {
		return TrainTestSplit{}, err
	}
	
	YTrain := selectSeriesRows(y, trainIndices)
	YTest := selectSeriesRows(y, testIndices)
	
	return TrainTestSplit{
		XTrain: XTrain,
		XTest:  XTest,
		YTrain: YTrain,
		YTest:  YTest,
	}, nil
}

// stratifiedSplit performs stratified sampling to maintain class proportions.
func stratifiedSplit(X *dataframe.DataFrame, y *seriesPkg.Series[any], testSize float64, rng *rand.Rand, stratifyCol string) (TrainTestSplit, error) {
	// Group indices by class
	classIndices := make(map[string][]int)
	
	for i := 0; i < y.Len(); i++ {
		val, ok := y.Get(i)
		if !ok || val == nil {
			continue
		}
		
		label := fmt.Sprint(val)
		classIndices[label] = append(classIndices[label], i)
	}
	
	trainIndices := make([]int, 0)
	testIndices := make([]int, 0)
	
	// Split each class proportionally
	for _, indices := range classIndices {
		// Shuffle class indices
		rng.Shuffle(len(indices), func(i, j int) {
			indices[i], indices[j] = indices[j], indices[i]
		})
		
		testN := int(float64(len(indices)) * testSize)
		trainN := len(indices) - testN
		
		trainIndices = append(trainIndices, indices[:trainN]...)
		testIndices = append(testIndices, indices[trainN:]...)
	}
	
	// Shuffle again to mix classes
	rng.Shuffle(len(trainIndices), func(i, j int) {
		trainIndices[i], trainIndices[j] = trainIndices[j], trainIndices[i]
	})
	rng.Shuffle(len(testIndices), func(i, j int) {
		testIndices[i], testIndices[j] = testIndices[j], testIndices[i]
	})
	
	// Create splits
	XTrain, err := selectRows(X, trainIndices)
	if err != nil {
		return TrainTestSplit{}, err
	}
	
	XTest, err := selectRows(X, testIndices)
	if err != nil {
		return TrainTestSplit{}, err
	}
	
	YTrain := selectSeriesRows(y, trainIndices)
	YTest := selectSeriesRows(y, testIndices)
	
	return TrainTestSplit{
		XTrain: XTrain,
		XTest:  XTest,
		YTrain: YTrain,
		YTest:  YTest,
	}, nil
}

// selectRows selects specific rows from a DataFrame by indices.
func selectRows(df *dataframe.DataFrame, indices []int) (*dataframe.DataFrame, error) {
	if len(indices) == 0 {
		// Return empty DataFrame with same structure
		emptyData := make(map[string]any)
		for _, col := range df.Columns() {
			emptyData[col] = []any{}
		}
		return dataframe.New(emptyData)
	}
	
	// Extract all columns
	cols := df.Columns()
	newData := make(map[string]any)
	
	for _, col := range cols {
		colSeries, err := df.Column(col)
		if err != nil {
			continue
		}
		
		colData := make([]any, len(indices))
		for i, idx := range indices {
			if idx < colSeries.Len() {
				val, ok := colSeries.Get(idx)
				if ok {
					colData[i] = val
				}
			}
		}
		newData[col] = colData
	}
	
	return dataframe.New(newData)
}

// selectSeriesRows selects specific rows from a Series by indices.
func selectSeriesRows(s *seriesPkg.Series[any], indices []int) *seriesPkg.Series[any] {
	data := make([]any, len(indices))
	
	for i, idx := range indices {
		if idx < s.Len() {
			if val, ok := s.Get(idx); ok {
				data[i] = val
			}
		}
	}
	
	return seriesPkg.New(s.Name(), data, s.Dtype())
}
