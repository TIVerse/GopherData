// Package crossval provides cross-validation utilities for model evaluation.
package crossval

import (
	"fmt"
	"math/rand"

	"github.com/TIVerse/GopherData/dataframe"
	seriesPkg "github.com/TIVerse/GopherData/series"
)

// Fold represents a single train/test split in cross-validation.
type Fold struct {
	TrainIndices []int
	TestIndices  []int
}

// KFold provides K-fold cross-validation splitting.
// Splits dataset into K consecutive folds. Each fold is used once as validation
// while the remaining K-1 folds form the training set.
type KFold struct {
	// NSplits is the number of folds (K)
	NSplits int
	
	// Shuffle whether to shuffle the data before splitting
	Shuffle bool
	
	// Seed for random number generator (used if Shuffle is true)
	Seed int64
}

// NewKFold creates a new K-fold cross-validator.
func NewKFold(nSplits int, shuffle bool, seed int64) *KFold {
	if nSplits < 2 {
		nSplits = 5
	}
	return &KFold{
		NSplits: nSplits,
		Shuffle: shuffle,
		Seed:    seed,
	}
}

// Split generates train/test index splits.
func (k *KFold) Split(df *dataframe.DataFrame) []Fold {
	n := df.Nrows()
	
	// Create indices
	indices := make([]int, n)
	for i := range indices {
		indices[i] = i
	}
	
	// Shuffle if requested
	if k.Shuffle {
		rng := rand.New(rand.NewSource(k.Seed))
		rng.Shuffle(len(indices), func(i, j int) {
			indices[i], indices[j] = indices[j], indices[i]
		})
	}
	
	// Calculate fold sizes
	foldSizes := make([]int, k.NSplits)
	baseSize := n / k.NSplits
	remainder := n % k.NSplits
	
	for i := 0; i < k.NSplits; i++ {
		foldSizes[i] = baseSize
		if i < remainder {
			foldSizes[i]++
		}
	}
	
	// Create folds
	folds := make([]Fold, k.NSplits)
	current := 0
	
	for i := 0; i < k.NSplits; i++ {
		// Test indices for this fold
		testStart := current
		testEnd := current + foldSizes[i]
		testIndices := make([]int, foldSizes[i])
		copy(testIndices, indices[testStart:testEnd])
		
		// Train indices are everything except this fold
		trainIndices := make([]int, 0, n-foldSizes[i])
		trainIndices = append(trainIndices, indices[:testStart]...)
		trainIndices = append(trainIndices, indices[testEnd:]...)
		
		folds[i] = Fold{
			TrainIndices: trainIndices,
			TestIndices:  testIndices,
		}
		
		current = testEnd
	}
	
	return folds
}

// CrossValScore performs cross-validation and returns scores for each fold.
type Scorer interface {
	Fit(X *dataframe.DataFrame, y *seriesPkg.Series[any]) error
	Predict(X *dataframe.DataFrame) (*seriesPkg.Series[any], error)
}

// CrossValScore evaluates a model using cross-validation.
// Returns a score for each fold.
func CrossValScore(model Scorer, X *dataframe.DataFrame, y *seriesPkg.Series[any], cv *KFold, scoringFunc func(*seriesPkg.Series[any], *seriesPkg.Series[any]) float64) ([]float64, error) {
	if X.Nrows() != y.Len() {
		return nil, fmt.Errorf("X and y must have same length")
	}
	
	folds := cv.Split(X)
	scores := make([]float64, len(folds))
	
	for i, fold := range folds {
		// Split data
		XTrain, err := selectRowsByIndices(X, fold.TrainIndices)
		if err != nil {
			return nil, fmt.Errorf("fold %d: failed to select train data: %w", i, err)
		}
		
		XTest, err := selectRowsByIndices(X, fold.TestIndices)
		if err != nil {
			return nil, fmt.Errorf("fold %d: failed to select test data: %w", i, err)
		}
		
		yTrain := selectSeriesByIndices(y, fold.TrainIndices)
		yTest := selectSeriesByIndices(y, fold.TestIndices)
		
		// Train model
		err = model.Fit(XTrain, yTrain)
		if err != nil {
			return nil, fmt.Errorf("fold %d: fit failed: %w", i, err)
		}
		
		// Predict
		yPred, err := model.Predict(XTest)
		if err != nil {
			return nil, fmt.Errorf("fold %d: predict failed: %w", i, err)
		}
		
		// Score
		scores[i] = scoringFunc(yTest, yPred)
	}
	
	return scores, nil
}

// Helper functions

func selectRowsByIndices(df *dataframe.DataFrame, indices []int) (*dataframe.DataFrame, error) {
	if len(indices) == 0 {
		emptyData := make(map[string]any)
		for _, col := range df.Columns() {
			emptyData[col] = []any{}
		}
		return dataframe.New(emptyData)
	}
	
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

func selectSeriesByIndices(s *seriesPkg.Series[any], indices []int) *seriesPkg.Series[any] {
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
