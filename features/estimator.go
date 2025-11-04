// Package features provides sklearn-style feature engineering transformers and pipelines.
package features

import (
	"github.com/TIVerse/GopherData/dataframe"
)

// Estimator learns from data and transforms it.
// This follows the sklearn fit/transform pattern.
type Estimator interface {
	// Fit learns parameters from the training data.
	// The target parameter is optional and used by supervised transformers (e.g., TargetEncoder).
	Fit(df *dataframe.DataFrame, target ...string) error

	// Transform applies the learned transformation to the data.
	// Returns a new DataFrame with the transformation applied.
	Transform(df *dataframe.DataFrame) (*dataframe.DataFrame, error)
}

// Transformer combines Fit and Transform operations.
// Most estimators implement this interface.
type Transformer interface {
	Estimator
	
	// FitTransform fits the estimator and transforms the data in one step.
	// This is often more efficient than calling Fit and Transform separately.
	FitTransform(df *dataframe.DataFrame, target ...string) (*dataframe.DataFrame, error)
}

// BaseFitTransform provides a default FitTransform implementation.
// Custom estimators can use this if they don't have an optimized FitTransform.
func BaseFitTransform(est Estimator, df *dataframe.DataFrame, target ...string) (*dataframe.DataFrame, error) {
	if err := est.Fit(df, target...); err != nil {
		return nil, err
	}
	return est.Transform(df)
}

// Fittable indicates whether an estimator has been fitted.
type Fittable interface {
	// IsFitted returns true if the estimator has been fitted.
	IsFitted() bool
}
