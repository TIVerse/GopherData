package decomposition

import (
	"math"
	"testing"

	"github.com/TIVerse/GopherData/dataframe"
)

func TestPCABasic(t *testing.T) {
	// Create correlated 2D data
	data := map[string]any{
		"x": []float64{1, 2, 3, 4, 5},
		"y": []float64{2, 4, 6, 8, 10}, // y = 2*x
	}
	X, _ := dataframe.New(data)
	
	pca := NewPCA(2)
	err := pca.Fit(X)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}
	
	explained := pca.ExplainedVarianceRatio()
	if len(explained) != 2 {
		t.Errorf("Expected 2 components, got %d", len(explained))
	}
	
	// First component should explain most of the variance
	if explained[0] < 0.9 {
		t.Errorf("Expected first component to explain >90%%, got %.2f%%", explained[0]*100)
	}
	
	t.Logf("Explained variance ratio: %.4f, %.4f", explained[0], explained[1])
	
	// Sum should be close to 1.0
	sum := explained[0] + explained[1]
	if math.Abs(sum-1.0) > 0.01 {
		t.Errorf("Explained variance should sum to ~1.0, got %.4f", sum)
	}
}

func TestPCATransform(t *testing.T) {
	data := map[string]any{
		"x": []float64{1, 2, 3, 4, 5},
		"y": []float64{1, 2, 3, 4, 5},
	}
	X, _ := dataframe.New(data)
	
	pca := NewPCA(2)
	pca.Fit(X)
	
	transformed, err := pca.Transform(X)
	if err != nil {
		t.Fatalf("Transform failed: %v", err)
	}
	
	if transformed.Nrows() != 5 {
		t.Errorf("Expected 5 rows, got %d", transformed.Nrows())
	}
	
	if transformed.Ncols() != 2 {
		t.Errorf("Expected 2 columns, got %d", transformed.Ncols())
	}
	
	// Check column names
	cols := transformed.Columns()
	if cols[0] != "PC1" || cols[1] != "PC2" {
		t.Errorf("Expected columns [PC1, PC2], got %v", cols)
	}
	
	t.Logf("Transformed data shape: %dx%d", transformed.Nrows(), transformed.Ncols())
}

func TestPCADimensionalityReduction(t *testing.T) {
	// 3D data reduced to 2D
	data := map[string]any{
		"x": []float64{1, 2, 3, 4, 5},
		"y": []float64{2, 4, 6, 8, 10},
		"z": []float64{1, 1, 1, 1, 1}, // Constant, no variance
	}
	X, _ := dataframe.New(data)
	
	pca := NewPCA(2)
	transformed, err := pca.FitTransform(X)
	if err != nil {
		t.Fatalf("FitTransform failed: %v", err)
	}
	
	if transformed.Ncols() != 2 {
		t.Errorf("Expected 2 principal components, got %d", transformed.Ncols())
	}
	
	// Should have kept most of the variance
	explained := pca.ExplainedVarianceRatio()
	totalExplained := 0.0
	for _, e := range explained {
		totalExplained += e
	}
	
	if totalExplained < 0.95 {
		t.Errorf("Expected to explain >95%% of variance, got %.2f%%", totalExplained*100)
	}
	
	t.Logf("Variance explained by 2 components: %.2f%%", totalExplained*100)
}

func TestPCAComponents(t *testing.T) {
	data := map[string]any{
		"x": []float64{1, 2, 3, 4, 5},
		"y": []float64{1, 2, 3, 4, 5},
	}
	X, _ := dataframe.New(data)
	
	pca := NewPCA(2)
	pca.Fit(X)
	
	components := pca.Components()
	if len(components) != 2 {
		t.Errorf("Expected 2 components, got %d", len(components))
	}
	
	// Each component should have 2 values (one per feature)
	for i, comp := range components {
		if len(comp) != 2 {
			t.Errorf("Component %d: expected length 2, got %d", i, len(comp))
		}
		
		// Components should be unit vectors (approximately)
		sum := 0.0
		for _, val := range comp {
			sum += val * val
		}
		norm := math.Sqrt(sum)
		
		if math.Abs(norm-1.0) > 0.01 {
			t.Errorf("Component %d: expected unit vector (norm=1), got norm=%.4f", i, norm)
		}
	}
	
	t.Logf("Component 1: %v", components[0])
	t.Logf("Component 2: %v", components[1])
}

func TestPCAFewerComponentsThanFeatures(t *testing.T) {
	// 5 features, keep only 2
	data := map[string]any{
		"f1": []float64{1, 2, 3, 4, 5},
		"f2": []float64{2, 4, 6, 8, 10},
		"f3": []float64{1, 1, 2, 2, 3},
		"f4": []float64{5, 4, 3, 2, 1},
		"f5": []float64{1, 2, 1, 2, 1},
	}
	X, _ := dataframe.New(data)
	
	pca := NewPCA(2)
	transformed, err := pca.FitTransform(X)
	if err != nil {
		t.Fatalf("FitTransform failed: %v", err)
	}
	
	if transformed.Ncols() != 2 {
		t.Errorf("Expected 2 components, got %d", transformed.Ncols())
	}
	
	explained := pca.ExplainedVarianceRatio()
	t.Logf("Explained variance (2 components from 5 features): %.2f%%, %.2f%%", 
		explained[0]*100, explained[1]*100)
}

func TestPCAExplainedVariance(t *testing.T) {
	data := map[string]any{
		"x": []float64{1, 2, 3, 4, 5},
		"y": []float64{10, 20, 30, 40, 50}, // 10x more variance but perfectly correlated with x
	}
	X, _ := dataframe.New(data)
	
	pca := NewPCA(2)
	pca.Fit(X)
	
	explained := pca.ExplainedVariance()
	
	// Check that first component has positive variance
	if explained[0] <= 0 {
		t.Error("First component should have positive variance")
	}
	
	// Second component may have zero variance for perfectly correlated data
	if explained[0] < explained[1] {
		t.Error("Explained variance should be in descending order")
	}
	
	t.Logf("Explained variance: %.4f, %.4f", explained[0], explained[1])
	t.Logf("Note: Second component is zero because x and y are perfectly correlated")
}
