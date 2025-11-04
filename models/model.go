// Package models provides machine learning algorithms and utilities.
package models

import (
	"github.com/TIVerse/GopherData/dataframe"
	seriesPkg "github.com/TIVerse/GopherData/series"
)

// Model is the base interface for all ML models.
type Model interface {
	// Fit trains the model on data X with target y
	Fit(X *dataframe.DataFrame, y *seriesPkg.Series[any]) error
	
	// Predict makes predictions on new data X
	Predict(X *dataframe.DataFrame) (*seriesPkg.Series[any], error)
}

// Classifier adds classification-specific methods.
type Classifier interface {
	Model
	
	// PredictProba returns class probabilities for each sample
	PredictProba(X *dataframe.DataFrame) (*dataframe.DataFrame, error)
}

// Regressor is a marker interface for regression models.
type Regressor interface {
	Model
}

// Clusterer is the interface for unsupervised clustering models.
type Clusterer interface {
	// Fit trains the clustering model on data X
	Fit(X *dataframe.DataFrame) error
	
	// Predict assigns cluster labels to new data X
	Predict(X *dataframe.DataFrame) (*seriesPkg.Series[any], error)
	
	// FitPredict fits the model and returns cluster labels
	FitPredict(X *dataframe.DataFrame) (*seriesPkg.Series[any], error)
}

// Transformer is the interface for dimensionality reduction and feature extraction.
type Transformer interface {
	// Fit learns the transformation from data X
	Fit(X *dataframe.DataFrame) error
	
	// Transform applies the learned transformation to X
	Transform(X *dataframe.DataFrame) (*dataframe.DataFrame, error)
	
	// FitTransform fits and transforms in one step
	FitTransform(X *dataframe.DataFrame) (*dataframe.DataFrame, error)
}
