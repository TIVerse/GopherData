// Package decomposition provides dimensionality reduction algorithms.
package decomposition

import (
	"fmt"

	"github.com/TIVerse/GopherData/core"
	"github.com/TIVerse/GopherData/dataframe"
	"gonum.org/v1/gonum/mat"
)

// PCA implements Principal Component Analysis for dimensionality reduction.
// PCA finds orthogonal directions of maximum variance in the data.
type PCA struct {
	// NComponents is the number of principal components to keep
	NComponents int
	
	// components stores the principal components (eigenvectors)
	// Shape: (n_components, n_features)
	components [][]float64
	
	// explainedVariance stores the variance explained by each component
	explainedVariance []float64
	
	// explainedVarianceRatio stores the percentage of variance explained
	explainedVarianceRatio []float64
	
	// mean stores the feature means (for centering)
	mean []float64
	
	// fitted indicates whether the model has been fitted
	fitted bool
	
	// featureNames stores the names of features
	featureNames []string
}

// NewPCA creates a new PCA model.
func NewPCA(nComponents int) *PCA {
	if nComponents < 1 {
		nComponents = 2
	}
	
	return &PCA{
		NComponents: nComponents,
		fitted:      false,
	}
}

// Fit learns the principal components from data X.
func (pca *PCA) Fit(X *dataframe.DataFrame) error {
	// Extract features
	features, names, err := extractFeaturesDecomp(X)
	if err != nil {
		return err
	}
	pca.featureNames = names
	
	n := len(features)
	p := len(features[0])
	
	if pca.NComponents > p {
		pca.NComponents = p
	}
	
	// Compute mean
	pca.mean = make([]float64, p)
	for j := 0; j < p; j++ {
		sum := 0.0
		for i := 0; i < n; i++ {
			sum += features[i][j]
		}
		pca.mean[j] = sum / float64(n)
	}
	
	// Center the data
	centered := make([][]float64, n)
	for i := range centered {
		centered[i] = make([]float64, p)
		for j := 0; j < p; j++ {
			centered[i][j] = features[i][j] - pca.mean[j]
		}
	}
	
	// Create data matrix
	XMat := mat.NewDense(n, p, nil)
	for i, row := range centered {
		XMat.SetRow(i, row)
	}
	
	// Compute covariance matrix: (1/n) * X^T * X
	var covDense mat.Dense
	covDense.Mul(XMat.T(), XMat)
	covDense.Scale(1.0/float64(n), &covDense)
	
	// Convert to symmetric matrix
	covSym := mat.NewSymDense(p, nil)
	for i := 0; i < p; i++ {
		for j := i; j < p; j++ {
			val := covDense.At(i, j)
			covSym.SetSym(i, j, val)
		}
	}
	
	// Eigenvalue decomposition
	var eig mat.EigenSym
	ok := eig.Factorize(covSym, true)
	if !ok {
		return fmt.Errorf("eigenvalue decomposition failed")
	}
	
	// Get eigenvalues and eigenvectors
	eigenvalues := make([]float64, p)
	eig.Values(eigenvalues)
	
	var eigenvectors mat.Dense
	eig.VectorsTo(&eigenvectors)
	
	// Sort by eigenvalue (descending)
	// Note: gonum returns them in ascending order
	pca.explainedVariance = make([]float64, pca.NComponents)
	pca.components = make([][]float64, pca.NComponents)
	
	totalVariance := 0.0
	for _, ev := range eigenvalues {
		totalVariance += ev
	}
	
	// Take the top NComponents (eigenvalues are in ascending order, so we start from the end)
	for i := 0; i < pca.NComponents; i++ {
		// Index from the end (largest eigenvalues)
		idx := p - 1 - i
		pca.explainedVariance[i] = eigenvalues[idx]
		
		// Extract eigenvector
		pca.components[i] = make([]float64, p)
		for j := 0; j < p; j++ {
			pca.components[i][j] = eigenvectors.At(j, idx)
		}
	}
	
	// Calculate explained variance ratio
	pca.explainedVarianceRatio = make([]float64, pca.NComponents)
	for i := 0; i < pca.NComponents; i++ {
		pca.explainedVarianceRatio[i] = pca.explainedVariance[i] / totalVariance
	}
	
	pca.fitted = true
	return nil
}

// Transform projects data onto the principal components.
func (pca *PCA) Transform(X *dataframe.DataFrame) (*dataframe.DataFrame, error) {
	if !pca.fitted {
		return nil, fmt.Errorf("model not fitted yet")
	}
	
	features, _, err := extractFeaturesDecomp(X)
	if err != nil {
		return nil, err
	}
	
	n := len(features)
	p := len(features[0])
	
	if p != len(pca.mean) {
		return nil, fmt.Errorf("feature count mismatch: expected %d, got %d", len(pca.mean), p)
	}
	
	// Center the data
	centered := make([][]float64, n)
	for i := range centered {
		centered[i] = make([]float64, p)
		for j := 0; j < p; j++ {
			centered[i][j] = features[i][j] - pca.mean[j]
		}
	}
	
	// Project onto principal components: X_transformed = X_centered * components^T
	transformed := make([][]float64, n)
	for i := range transformed {
		transformed[i] = make([]float64, pca.NComponents)
		for j := 0; j < pca.NComponents; j++ {
			sum := 0.0
			for k := 0; k < p; k++ {
				sum += centered[i][k] * pca.components[j][k]
			}
			transformed[i][j] = sum
		}
	}
	
	// Create result DataFrame
	resultData := make(map[string]any)
	for j := 0; j < pca.NComponents; j++ {
		colName := fmt.Sprintf("PC%d", j+1)
		colData := make([]any, n)
		for i := 0; i < n; i++ {
			colData[i] = transformed[i][j]
		}
		resultData[colName] = colData
	}
	
	return dataframe.New(resultData)
}

// FitTransform fits the model and transforms the data in one step.
func (pca *PCA) FitTransform(X *dataframe.DataFrame) (*dataframe.DataFrame, error) {
	err := pca.Fit(X)
	if err != nil {
		return nil, err
	}
	return pca.Transform(X)
}

// ExplainedVariance returns the variance explained by each component.
func (pca *PCA) ExplainedVariance() []float64 {
	return pca.explainedVariance
}

// ExplainedVarianceRatio returns the proportion of variance explained by each component.
func (pca *PCA) ExplainedVarianceRatio() []float64 {
	return pca.explainedVarianceRatio
}

// Components returns the principal components (eigenvectors).
func (pca *PCA) Components() [][]float64 {
	return pca.components
}

// Helper functions

func extractFeaturesDecomp(X *dataframe.DataFrame) ([][]float64, []string, error) {
	n := X.Nrows()
	cols := X.Columns()
	
	// Filter numeric columns
	numericCols := make([]string, 0)
	for _, col := range cols {
		series, err := X.Column(col)
		if err != nil {
			continue
		}
		if isNumericDtypeDecomp(series.Dtype()) {
			numericCols = append(numericCols, col)
		}
	}
	
	if len(numericCols) == 0 {
		return nil, nil, fmt.Errorf("no numeric columns found")
	}
	
	features := make([][]float64, n)
	for i := range features {
		features[i] = make([]float64, len(numericCols))
	}
	
	for j, col := range numericCols {
		series, _ := X.Column(col)
		for i := 0; i < n; i++ {
			val, ok := series.Get(i)
			if ok && val != nil {
				features[i][j] = toFloat64Decomp(val)
			}
		}
	}
	
	return features, numericCols, nil
}

func isNumericDtypeDecomp(dtype core.Dtype) bool {
	return dtype == core.DtypeFloat64 || dtype == core.DtypeInt64
}

func toFloat64Decomp(val any) float64 {
	switch v := val.(type) {
	case float64:
		return v
	case float32:
		return float64(v)
	case int:
		return float64(v)
	case int64:
		return float64(v)
	case int32:
		return float64(v)
	default:
		return 0
	}
}
