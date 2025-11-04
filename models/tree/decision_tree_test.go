package tree

import (
	"testing"

	"github.com/TIVerse/GopherData/core"
	"github.com/TIVerse/GopherData/dataframe"
	seriesPkg "github.com/TIVerse/GopherData/series"
)

func TestDecisionTreeClassifier(t *testing.T) {
	// Simple classification data (XOR problem approximation)
	data := map[string]any{
		"x1": []float64{0, 0, 1, 1, 0, 0, 1, 1},
		"x2": []float64{0, 1, 0, 1, 0, 1, 0, 1},
	}
	X, _ := dataframe.New(data)
	
	yData := []any{"A", "B", "B", "A", "A", "B", "B", "A"}
	y := seriesPkg.New("y", yData, core.DtypeString)
	
	model := NewDecisionTreeClassifier(5, 2, "gini")
	err := model.Fit(X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}
	
	predictions, err := model.Predict(X)
	if err != nil {
		t.Fatalf("Predict failed: %v", err)
	}
	
	if predictions.Len() != 8 {
		t.Errorf("Expected 8 predictions, got %d", predictions.Len())
	}
	
	// Check accuracy
	correct := 0
	for i := 0; i < predictions.Len(); i++ {
		pred, _ := predictions.Get(i)
		true_, _ := y.Get(i)
		if pred == true_ {
			correct++
		}
	}
	
	accuracy := float64(correct) / float64(predictions.Len())
	if accuracy < 0.5 {
		t.Errorf("Expected accuracy > 0.5, got %.2f", accuracy)
	}
	
	t.Logf("Classification accuracy: %.2f", accuracy)
}

func TestDecisionTreeRegressor(t *testing.T) {
	// Simple regression data
	data := map[string]any{
		"x": []float64{1, 2, 3, 4, 5, 6, 7, 8},
	}
	X, _ := dataframe.New(data)
	
	yData := []any{2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0}
	y := seriesPkg.New("y", yData, core.DtypeFloat64)
	
	model := NewDecisionTreeRegressor(5, 2)
	err := model.Fit(X, y)
	if err != nil{
		t.Fatalf("Fit failed: %v", err)
	}
	
	predictions, err := model.Predict(X)
	if err != nil {
		t.Fatalf("Predict failed: %v", err)
	}
	
	if predictions.Len() != 8 {
		t.Errorf("Expected 8 predictions, got %d", predictions.Len())
	}
	
	t.Log("Regression test passed")
}

func TestDecisionTreeFeatureImportances(t *testing.T) {
	data := map[string]any{
		"x1": []float64{1, 2, 3, 4, 5},
		"x2": []float64{1, 1, 1, 1, 1}, // Constant, no importance
	}
	X, _ := dataframe.New(data)
	
	yData := []any{"A", "A", "B", "B", "B"}
	y := seriesPkg.New("y", yData, core.DtypeString)
	
	model := NewDecisionTreeClassifier(5, 2, "gini")
	model.Fit(X, y)
	
	importances := model.FeatureImportances()
	if len(importances) != 2 {
		t.Errorf("Expected 2 feature importances, got %d", len(importances))
	}
	
	t.Logf("Feature importances: %v", importances)
}

func TestDecisionTreeEntropy(t *testing.T) {
	data := map[string]any{
		"x": []float64{1, 2, 3, 4, 5, 6},
	}
	X, _ := dataframe.New(data)
	
	yData := []any{"A", "A", "A", "B", "B", "B"}
	y := seriesPkg.New("y", yData, core.DtypeString)
	
	model := NewDecisionTreeClassifier(5, 2, "entropy")
	err := model.Fit(X, y)
	if err != nil {
		t.Fatalf("Fit with entropy failed: %v", err)
	}
	
	predictions, err := model.Predict(X)
	if err != nil {
		t.Fatalf("Predict failed: %v", err)
	}
	
	if predictions.Len() != 6 {
		t.Errorf("Expected 6 predictions, got %d", predictions.Len())
	}
	
	t.Log("Entropy criterion test passed")
}
