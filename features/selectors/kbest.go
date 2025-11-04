package selectors

import (
	"fmt"
	"sort"

	"github.com/TIVerse/GopherData/dataframe"
)

// SelectKBest selects the K best features based on a scoring function.
// This is a simplified implementation using correlation with target.
type SelectKBest struct {
	// K is the number of features to select
	K int
	
	// Selected column names
	selected []string
	scores   map[string]float64
	fitted   bool
}

// NewSelectKBest creates a new SelectKBest selector.
func NewSelectKBest(k int) *SelectKBest {
	return &SelectKBest{
		K:        k,
		selected: nil,
		scores:   make(map[string]float64),
		fitted:   false,
	}
}

// Fit computes scores for each feature and selects top K.
func (s *SelectKBest) Fit(df *dataframe.DataFrame, target ...string) error {
	if len(target) == 0 {
		return fmt.Errorf("target column required for SelectKBest")
	}
	targetCol := target[0]
	
	targetSeries, err := df.Column(targetCol)
	if err != nil {
		return fmt.Errorf("target column %q: %w", targetCol, err)
	}
	
	cols := getNumericColumns(df)
	if len(cols) == 0 {
		return fmt.Errorf("no numeric columns to select from")
	}
	
	// Compute correlation score with target for each feature
	s.scores = make(map[string]float64)
	for _, col := range cols {
		if col == targetCol {
			continue
		}
		
		series, _ := df.Column(col)
		score := computeCorrelation(series, targetSeries)
		s.scores[col] = score
	}
	
	// Select top K features
	s.selected = selectTopK(s.scores, s.K)
	s.fitted = true
	return nil
}

// Transform returns a DataFrame with only the K best features.
func (s *SelectKBest) Transform(df *dataframe.DataFrame) (*dataframe.DataFrame, error) {
	if !s.fitted {
		return nil, fmt.Errorf("selector not fitted")
	}
	
	return df.Select(s.selected...), nil
}

// FitTransform fits the selector and transforms the data in one step.
func (s *SelectKBest) FitTransform(df *dataframe.DataFrame, target ...string) (*dataframe.DataFrame, error) {
	if err := s.Fit(df, target...); err != nil {
		return nil, err
	}
	return s.Transform(df)
}

// IsFitted returns true if the selector has been fitted.
func (s *SelectKBest) IsFitted() bool {
	return s.fitted
}

// GetSelectedFeatures returns the list of selected feature names.
func (s *SelectKBest) GetSelectedFeatures() []string {
	selected := make([]string, len(s.selected))
	copy(selected, s.selected)
	return selected
}

// GetScores returns the computed scores for each feature.
func (s *SelectKBest) GetScores() map[string]float64 {
	scores := make(map[string]float64, len(s.scores))
	for k, v := range s.scores {
		scores[k] = v
	}
	return scores
}

func computeCorrelation(x, y interface{ Len() int; Get(int) (any, bool) }) float64 {
	// Simplified Pearson correlation
	n := x.Len()
	if n != y.Len() {
		return 0
	}
	
	var sumX, sumY, sumXY, sumX2, sumY2 float64
	count := 0
	
	for i := 0; i < n; i++ {
		xVal, xOk := x.Get(i)
		yVal, yOk := y.Get(i)
		if !xOk || !yOk {
			continue
		}
		
		xFloat := toFloat64Selector(xVal)
		yFloat := toFloat64Selector(yVal)
		
		sumX += xFloat
		sumY += yFloat
		sumXY += xFloat * yFloat
		sumX2 += xFloat * xFloat
		sumY2 += yFloat * yFloat
		count++
	}
	
	if count < 2 {
		return 0
	}
	
	numerator := float64(count)*sumXY - sumX*sumY
	denominatorX := float64(count)*sumX2 - sumX*sumX
	denominatorY := float64(count)*sumY2 - sumY*sumY
	
	if denominatorX == 0 || denominatorY == 0 {
		return 0
	}
	
	return numerator / (denominatorX * denominatorY)
}

func selectTopK(scores map[string]float64, k int) []string {
	type featureScore struct {
		feature string
		score   float64
	}
	
	// Convert to slice
	features := make([]featureScore, 0, len(scores))
	for feature, score := range scores {
		features = append(features, featureScore{feature, score})
	}
	
	// Sort by score (descending)
	sort.Slice(features, func(i, j int) bool {
		return features[i].score > features[j].score
	})
	
	// Take top K
	if k > len(features) {
		k = len(features)
	}
	
	selected := make([]string, k)
	for i := 0; i < k; i++ {
		selected[i] = features[i].feature
	}
	
	return selected
}
