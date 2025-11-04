package crossval

import (
	"fmt"
	"math/rand"

	"github.com/TIVerse/GopherData/dataframe"
	seriesPkg "github.com/TIVerse/GopherData/series"
)

// StratifiedKFold provides stratified K-fold cross-validation.
// Stratification ensures that each fold maintains the same class distribution
// as the complete dataset. Important for imbalanced datasets.
type StratifiedKFold struct {
	// NSplits is the number of folds
	NSplits int
	
	// Shuffle whether to shuffle within each class before splitting
	Shuffle bool
	
	// Seed for random number generator
	Seed int64
}

// NewStratifiedKFold creates a new stratified K-fold cross-validator.
func NewStratifiedKFold(nSplits int, shuffle bool, seed int64) *StratifiedKFold {
	if nSplits < 2 {
		nSplits = 5
	}
	return &StratifiedKFold{
		NSplits: nSplits,
		Shuffle: shuffle,
		Seed:    seed,
	}
}

// Split generates stratified train/test splits.
// The y parameter is used for stratification (typically class labels).
func (s *StratifiedKFold) Split(X *dataframe.DataFrame, y *seriesPkg.Series[any]) []Fold {
	if X.Nrows() != y.Len() {
		return nil
	}
	
	n := X.Nrows()
	
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
	
	// Shuffle within each class if requested
	if s.Shuffle {
		rng := rand.New(rand.NewSource(s.Seed))
		for _, indices := range classIndices {
			rng.Shuffle(len(indices), func(i, j int) {
				indices[i], indices[j] = indices[j], indices[i]
			})
		}
	}
	
	// Split each class into folds
	classFolds := make(map[string][][]int)
	for label, indices := range classIndices {
		classFolds[label] = make([][]int, s.NSplits)
		
		// Distribute indices across folds
		for i, idx := range indices {
			foldIdx := i % s.NSplits
			classFolds[label][foldIdx] = append(classFolds[label][foldIdx], idx)
		}
	}
	
	// Combine folds from all classes
	folds := make([]Fold, s.NSplits)
	
	for i := 0; i < s.NSplits; i++ {
		testIndices := make([]int, 0)
		trainIndices := make([]int, 0, n)
		
		// Collect test indices from all classes for this fold
		for _, classFold := range classFolds {
			testIndices = append(testIndices, classFold[i]...)
		}
		
		// Collect train indices from all other folds
		for j := 0; j < s.NSplits; j++ {
			if j != i {
				for _, classFold := range classFolds {
					trainIndices = append(trainIndices, classFold[j]...)
				}
			}
		}
		
		folds[i] = Fold{
			TrainIndices: trainIndices,
			TestIndices:  testIndices,
		}
	}
	
	return folds
}

// CrossValScoreStratified performs stratified cross-validation.
func CrossValScoreStratified(model Scorer, X *dataframe.DataFrame, y *seriesPkg.Series[any], cv *StratifiedKFold, scoringFunc func(*seriesPkg.Series[any], *seriesPkg.Series[any]) float64) ([]float64, error) {
	if X.Nrows() != y.Len() {
		return nil, fmt.Errorf("x and y must have same length")
	}
	
	folds := cv.Split(X, y)
	if folds == nil {
		return nil, fmt.Errorf("failed to create folds")
	}
	
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
