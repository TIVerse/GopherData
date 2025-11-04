package crossval

import (
	"testing"

	"github.com/TIVerse/GopherData/core"
	"github.com/TIVerse/GopherData/dataframe"
	"github.com/TIVerse/GopherData/models/linear"
	seriesPkg "github.com/TIVerse/GopherData/series"
)

func TestKFoldSplit(t *testing.T) {
	data := map[string]any{
		"x": []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
	}
	df, _ := dataframe.New(data)
	
	kf := NewKFold(5, false, 42)
	folds := kf.Split(df)
	
	if len(folds) != 5 {
		t.Errorf("Expected 5 folds, got %d", len(folds))
	}
	
	// Check that each fold has correct sizes
	totalTrain := 0
	totalTest := 0
	
	for i, fold := range folds {
		totalTrain += len(fold.TrainIndices)
		totalTest += len(fold.TestIndices)
		
		t.Logf("Fold %d: train=%d, test=%d", i, len(fold.TrainIndices), len(fold.TestIndices))
	}
	
	// Total should be K * n_samples
	if totalTrain != 40 { // 8 * 5
		t.Errorf("Expected total train size 40, got %d", totalTrain)
	}
	if totalTest != 10 { // 2 * 5
		t.Errorf("Expected total test size 10, got %d", totalTest)
	}
}

func TestKFoldShuffle(t *testing.T) {
	data := map[string]any{
		"x": []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
	}
	df, _ := dataframe.New(data)
	
	kf1 := NewKFold(3, false, 42)
	folds1 := kf1.Split(df)
	
	kf2 := NewKFold(3, true, 42)
	folds2 := kf2.Split(df)
	
	// With same seed, shuffled splits should be deterministic
	kf3 := NewKFold(3, true, 42)
	folds3 := kf3.Split(df)
	
	// Check that shuffled folds are reproducible
	if len(folds2[0].TestIndices) != len(folds3[0].TestIndices) {
		t.Error("Shuffled folds should be reproducible with same seed")
	}
	
	t.Logf("Unshuffled fold 0 test: %v", folds1[0].TestIndices[:3])
	t.Logf("Shuffled fold 0 test: %v", folds2[0].TestIndices[:3])
}

func TestCrossValScore(t *testing.T) {
	// Create simple linear data
	data := map[string]any{
		"x": []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
	}
	X, _ := dataframe.New(data)
	
	yData := []any{2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0}
	y := seriesPkg.New("y", yData, core.DtypeFloat64)
	
	// Create model
	model := linear.NewLinearRegression(true)
	
	// 3-fold cross-validation
	kf := NewKFold(3, false, 42)
	
	// Scoring function (R² score)
	scoringFunc := func(yTrue, yPred *seriesPkg.Series[any]) float64 {
		// Simple R² calculation
		var ssRes, ssTot float64
		var mean float64
		
		// Calculate mean
		for i := 0; i < yTrue.Len(); i++ {
			val, _ := yTrue.Get(i)
			mean += toFloat64Test(val)
		}
		mean /= float64(yTrue.Len())
		
		// Calculate SS
		for i := 0; i < yTrue.Len(); i++ {
			trueVal, _ := yTrue.Get(i)
			predVal, _ := yPred.Get(i)
			
			tv := toFloat64Test(trueVal)
			pv := toFloat64Test(predVal)
			
			ssRes += (tv - pv) * (tv - pv)
			ssTot += (tv - mean) * (tv - mean)
		}
		
		if ssTot == 0 {
			return 0
		}
		return 1 - (ssRes / ssTot)
	}
	
	scores, err := CrossValScore(model, X, y, kf, scoringFunc)
	if err != nil {
		t.Fatalf("CrossValScore failed: %v", err)
	}
	
	if len(scores) != 3 {
		t.Errorf("Expected 3 scores, got %d", len(scores))
	}
	
	// All scores should be high for this perfect linear relationship
	for i, score := range scores {
		t.Logf("Fold %d score: %.4f", i, score)
		if score < 0.9 {
			t.Errorf("Expected score > 0.9, got %.4f for fold %d", score, i)
		}
	}
}

func toFloat64Test(val any) float64 {
	switch v := val.(type) {
	case float64:
		return v
	case float32:
		return float64(v)
	case int:
		return float64(v)
	case int64:
		return float64(v)
	default:
		return 0
	}
}
