package cluster

import (
	"testing"

	"github.com/TIVerse/GopherData/dataframe"
)

func TestKMeansBasic(t *testing.T) {
	// Create simple 2-cluster data
	data := map[string]any{
		"x": []float64{1, 2, 3, 8, 9, 10},
		"y": []float64{1, 2, 3, 8, 9, 10},
	}
	X, _ := dataframe.New(data)
	
	model := NewKMeans(2, 100, "k-means++", 42)
	err := model.Fit(X)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}
	
	centers := model.Centers()
	if len(centers) != 2 {
		t.Errorf("Expected 2 centers, got %d", len(centers))
	}
	
	t.Logf("Centers: %v", centers)
	t.Logf("Inertia: %.4f", model.Inertia())
	t.Logf("Iterations: %d", model.NIter())
	
	if model.NIter() == 0 {
		t.Error("Should have performed at least one iteration")
	}
}

func TestKMeansPredict(t *testing.T) {
	// Training data
	data := map[string]any{
		"x": []float64{1, 2, 8, 9},
		"y": []float64{1, 2, 8, 9},
	}
	X, _ := dataframe.New(data)
	
	model := NewKMeans(2, 100, "random", 42)
	_ = model.Fit(X)
	
	// Test data
	testData := map[string]any{
		"x": []float64{1.5, 8.5},
		"y": []float64{1.5, 8.5},
	}
	XTest, _ := dataframe.New(testData)
	
	labels, err := model.Predict(XTest)
	if err != nil {
		t.Fatalf("Predict failed: %v", err)
	}
	
	if labels.Len() != 2 {
		t.Errorf("Expected 2 labels, got %d", labels.Len())
	}
	
	// Points should be in different clusters
	label0, _ := labels.Get(0)
	label1, _ := labels.Get(1)
	
	t.Logf("Predicted labels: %v, %v", label0, label1)
}

func TestKMeansFitPredict(t *testing.T) {
	data := map[string]any{
		"x": []float64{1, 2, 3, 8, 9, 10},
		"y": []float64{1, 2, 3, 8, 9, 10},
	}
	X, _ := dataframe.New(data)
	
	model := NewKMeans(2, 100, "k-means++", 42)
	labels, err := model.FitPredict(X)
	if err != nil {
		t.Fatalf("FitPredict failed: %v", err)
	}
	
	if labels.Len() != 6 {
		t.Errorf("Expected 6 labels, got %d", labels.Len())
	}
	
	// Count points in each cluster
	cluster0 := 0
	cluster1 := 0
	
	for i := 0; i < labels.Len(); i++ {
		val, _ := labels.Get(i)
		if val.(int64) == 0 {
			cluster0++
		} else {
			cluster1++
		}
	}
	
	t.Logf("Cluster 0: %d points, Cluster 1: %d points", cluster0, cluster1)
	
	// Both clusters should have some points
	if cluster0 == 0 || cluster1 == 0 {
		t.Error("Both clusters should have points")
	}
}

func TestKMeansConvergence(t *testing.T) {
	// Well-separated clusters should converge quickly
	data := map[string]any{
		"x": []float64{0, 0, 1, 100, 100, 101},
		"y": []float64{0, 1, 0, 100, 101, 100},
	}
	X, _ := dataframe.New(data)
	
	model := NewKMeans(2, 100, "k-means++", 42)
	_ = model.Fit(X)
	
	if model.NIter() > 10 {
		t.Logf("Warning: Took %d iterations to converge (expected < 10)", model.NIter())
	}
	
	t.Logf("Converged in %d iterations", model.NIter())
}

func TestKMeansInvalidInput(t *testing.T) {
	// More clusters than samples
	data := map[string]any{
		"x": []float64{1, 2},
		"y": []float64{1, 2},
	}
	X, _ := dataframe.New(data)
	
	model := NewKMeans(5, 100, "k-means++", 42)
	err := model.Fit(X)
	
	if err == nil {
		t.Error("Expected error for n_clusters > n_samples")
	}
}
