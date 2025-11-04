package selectors

import (
	"fmt"
	"math"
	"sort"

	"github.com/TIVerse/GopherData/dataframe"
)

// RFE (Recursive Feature Elimination) selects features by recursively removing the least important features.
// Uses correlation with target and mutual feature redundancy for importance scoring.
type RFE struct {
	// NFeatures is the number of features to select
	NFeatures int
	
	// Step is the number of features to remove at each iteration
	Step int
	
	// Selected column names
	selected []string
	
	// Feature rankings (1 = best)
	rankings map[string]int
	
	fitted bool
}

// NewRFE creates a new RFE selector.
func NewRFE(nFeatures int) *RFE {
	return &RFE{
		NFeatures: nFeatures,
		Step:      1,
		selected:  nil,
		rankings:  make(map[string]int),
		fitted:    false,
	}
}

// Fit performs recursive feature elimination using correlation-based importance.
func (r *RFE) Fit(df *dataframe.DataFrame, target ...string) error {
	cols := getNumericColumns(df)
	if len(cols) == 0 {
		return fmt.Errorf("no numeric columns to select from")
	}
	
	// If we want more features than available, select all
	if r.NFeatures >= len(cols) {
		r.selected = cols
		for i, col := range cols {
			r.rankings[col] = i + 1
		}
		r.fitted = true
		return nil
	}
	
	// Validate target if provided
	var targetCol string
	if len(target) > 0 {
		targetCol = target[0]
		if !df.HasColumn(targetCol) {
			return fmt.Errorf("target column %q not found", targetCol)
		}
	}
	
	// Start with all features
	remaining := make([]string, len(cols))
	copy(remaining, cols)
	rank := len(cols)
	
	// Recursively eliminate features
	for len(remaining) > r.NFeatures {
		// Compute importance scores for remaining features
		scores := r.computeImportanceScores(df, remaining, targetCol)
		
		// Sort by importance (ascending - least important first)
		type featureScore struct {
			name  string
			score float64
		}
		featureScores := make([]featureScore, 0, len(remaining))
		for _, col := range remaining {
			featureScores = append(featureScores, featureScore{col, scores[col]})
		}
		sort.Slice(featureScores, func(i, j int) bool {
			return featureScores[i].score < featureScores[j].score
		})
		
		// Remove the least important features (up to Step features)
		numToRemove := r.Step
		if len(remaining)-numToRemove < r.NFeatures {
			numToRemove = len(remaining) - r.NFeatures
		}
		
		// Assign rankings to removed features
		for i := 0; i < numToRemove; i++ {
			r.rankings[featureScores[i].name] = rank
			rank--
		}
		
		// Update remaining features
		newRemaining := make([]string, 0, len(remaining)-numToRemove)
		for i := numToRemove; i < len(featureScores); i++ {
			newRemaining = append(newRemaining, featureScores[i].name)
		}
		remaining = newRemaining
	}
	
	// Assign rankings to selected features
	for i, col := range remaining {
		r.rankings[col] = i + 1
	}
	
	r.selected = remaining
	r.fitted = true
	return nil
}

// computeImportanceScores computes feature importance using correlation with target
// and redundancy with other features
func (r *RFE) computeImportanceScores(df *dataframe.DataFrame, features []string, targetCol string) map[string]float64 {
	scores := make(map[string]float64)
	
	for _, feature := range features {
		var importance float64
		
		if targetCol != "" && targetCol != feature {
			// Use correlation with target as importance
			correlation := computeCorrelationRFE(df, feature, targetCol)
			importance = math.Abs(correlation)
		} else {
			// Use variance as importance when no target
			series, _ := df.Column(feature)
			importance = computeVariance(series)
		}
		
		// Add redundancy penalty (correlation with other features)
		redundancy := 0.0
		count := 0
		for _, other := range features {
			if other != feature {
				corr := computeCorrelationRFE(df, feature, other)
				redundancy += math.Abs(corr)
				count++
			}
		}
		if count > 0 {
			redundancy /= float64(count)
		}
		
		// Final score: importance minus redundancy penalty
		scores[feature] = importance - 0.3*redundancy
	}
	
	return scores
}

// computeCorrelationRFE computes Pearson correlation between two columns
func computeCorrelationRFE(df *dataframe.DataFrame, col1, col2 string) float64 {
	series1, err1 := df.Column(col1)
	series2, err2 := df.Column(col2)
	if err1 != nil || err2 != nil {
		return 0
	}
	
	// Collect valid pairs
	var x, y []float64
	for i := 0; i < series1.Len(); i++ {
		v1, ok1 := series1.Get(i)
		v2, ok2 := series2.Get(i)
		if ok1 && ok2 {
			x = append(x, toFloat64RFE(v1))
			y = append(y, toFloat64RFE(v2))
		}
	}
	
	if len(x) < 2 {
		return 0
	}
	
	// Compute means
	var sumX, sumY float64
	for i := range x {
		sumX += x[i]
		sumY += y[i]
	}
	meanX := sumX / float64(len(x))
	meanY := sumY / float64(len(y))
	
	// Compute correlation
	var numerator, denomX, denomY float64
	for i := range x {
		dx := x[i] - meanX
		dy := y[i] - meanY
		numerator += dx * dy
		denomX += dx * dx
		denomY += dy * dy
	}
	
	if denomX == 0 || denomY == 0 {
		return 0
	}
	
	return numerator / math.Sqrt(denomX*denomY)
}

func toFloat64RFE(val any) float64 {
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

// Transform returns a DataFrame with selected features.
func (r *RFE) Transform(df *dataframe.DataFrame) (*dataframe.DataFrame, error) {
	if !r.fitted {
		return nil, fmt.Errorf("selector not fitted")
	}
	
	return df.Select(r.selected...), nil
}

// FitTransform fits the selector and transforms the data in one step.
func (r *RFE) FitTransform(df *dataframe.DataFrame, target ...string) (*dataframe.DataFrame, error) {
	if err := r.Fit(df, target...); err != nil {
		return nil, err
	}
	return r.Transform(df)
}

// IsFitted returns true if the selector has been fitted.
func (r *RFE) IsFitted() bool {
	return r.fitted
}

// GetSelectedFeatures returns the list of selected feature names.
func (r *RFE) GetSelectedFeatures() []string {
	selected := make([]string, len(r.selected))
	copy(selected, r.selected)
	return selected
}
