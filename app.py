import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Set page config 
st.set_page_config(page_title="Breast Cancer Decision Tree", layout="wide")

# Function to split data
def stratified_split(X, y, test_size=0.2, random_state=42):
    """Stratified split for numpy arrays"""
    np.random.seed(random_state)
    
    if hasattr(X, 'values'):
        X = X.values
    if hasattr(y, 'values'):
        y = y.values
    
    classes = np.unique(y)
    train_indices = []
    test_indices = []
    
    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        np.random.shuffle(cls_indices)
        split_point = int(len(cls_indices) * (1 - test_size))
        train_indices.extend(cls_indices[:split_point])
        test_indices.extend(cls_indices[split_point:])
    
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test

# Helper function to count nodes
def count_nodes(node):
    """Count total nodes in tree"""
    if node['type'] == 'leaf':
        return 1
    else:
        return 1 + count_nodes(node['left']) + count_nodes(node['right'])
    
# Decision Tree Class
class SimpleDecisionTree:
    """A decision tree for classification (works with any numerical dataset)"""
    
    def __init__(self, max_depth=5, min_samples_split=2, min_impurity_decrease=0.0):
        """
        Parameters:
        - max_depth: Maximum depth of the tree
        - min_samples_split: Minimum samples required to split a node
        - min_impurity_decrease: Minimum gain required to split
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.tree = None  # Will store the tree structure
        
    def impurity(self, y):
        """Calculate impurity (1 - max class probability)"""
        if len(y) == 0:
            return 0
        # Using majority class impurity
        most_common = np.bincount(y.astype(int)).max()
        return 1 - (most_common / len(y))
    
    def gini_impurity(self, y):
        """Gini impurity (more common in decision trees)"""
        if len(y) == 0:
            return 0
        classes = np.unique(y)
        prob = [np.sum(y == c) / len(y) for c in classes]
        return 1 - sum(p**2 for p in prob)
    
    def entropy(self, y):
        """Entropy (alternative impurity measure)"""
        if len(y) == 0:
            return 0
        classes = np.unique(y)
        prob = [np.sum(y == c) / len(y) for c in classes]
        return -sum(p * np.log2(p) for p in prob if p > 0)
    
    def information_gain(self, y_parent, y_left, y_right, method='gini'):
        """Calculate information gain from split"""
        if method == 'gini':
            parent_impurity = self.gini_impurity(y_parent)
            left_impurity = self.gini_impurity(y_left)
            right_impurity = self.gini_impurity(y_right)
        elif method == 'entropy':
            parent_impurity = self.entropy(y_parent)
            left_impurity = self.entropy(y_left)
            right_impurity = self.entropy(y_right)
        else:
            parent_impurity = self.impurity(y_parent)
            left_impurity = self.impurity(y_left)
            right_impurity = self.impurity(y_right)
        
        n_parent = len(y_parent)
        n_left = len(y_left)
        n_right = len(y_right)
        
        weighted_impurity = (n_left/n_parent)*left_impurity + (n_right/n_parent)*right_impurity
        
        return parent_impurity - weighted_impurity
    
    def find_best_split(self, X, y):
        """Find the best split for the current node"""
        best_gain = 0
        best_split = None
        n_samples, n_features = X.shape
        
        # Check if we should split
        if len(y) < self.min_samples_split:
            return None
        
        # Try each feature
        for feature_idx in range(n_features):
            # Get unique values for this feature
            thresholds = np.unique(X[:, feature_idx])
            
            # Try each threshold (for efficiency, sample if too many)
            if len(thresholds) > 100:
                thresholds = np.percentile(X[:, feature_idx], np.linspace(0, 100, 50))
            
            for threshold in thresholds:
                # Split the data
                left_mask = X[:, feature_idx] < threshold
                right_mask = ~left_mask
                
                # Skip if split is invalid
                if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
                    continue
                
                # Calculate gain
                gain = self.information_gain(y, y[left_mask], y[right_mask])
                
                # Update best split
                if gain > best_gain:
                    best_gain = gain
                    best_split = {
                        'feature_idx': feature_idx,
                        'threshold': threshold,
                        'gain': gain,
                        'left_mask': left_mask,
                        'right_mask': right_mask
                    }
        
        # Only return split if gain is significant
        if best_split and best_split['gain'] > self.min_impurity_decrease:
            return best_split
        return None
    
    def build_tree(self, X, y, depth=0):
        """Recursively build the decision tree"""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Stopping conditions
        if (depth >= self.max_depth or 
            n_classes == 1 or 
            n_samples < self.min_samples_split):
            # Return leaf node with majority class
            return {
                'type': 'leaf',
                'prediction': np.bincount(y.astype(int)).argmax(),
                'n_samples': n_samples,
                'class_distribution': dict(Counter(y))
            }
        
        # Find best split
        best_split = self.find_best_split(X, y)
        
        if best_split is None:
            # No good split found, make leaf
            return {
                'type': 'leaf',
                'prediction': np.bincount(y.astype(int)).argmax(),
                'n_samples': n_samples,
                'class_distribution': dict(Counter(y))
            }
        
        # Split the data
        left_mask = best_split['left_mask']
        right_mask = best_split['right_mask']
        
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]
        
        # Recursively build left and right subtrees
        return {
            'type': 'split',
            'feature_idx': best_split['feature_idx'],
            'threshold': best_split['threshold'],
            'gain': best_split['gain'],
            'left': self.build_tree(X_left, y_left, depth + 1),
            'right': self.build_tree(X_right, y_right, depth + 1),
            'n_samples': n_samples,
            'class_distribution': dict(Counter(y))
        }
    
    def fit(self, X, y):
        """Train the decision tree"""
        # Convert to numpy arrays
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
        
        # Ensure y is integer type
        y = y.astype(int)
        
        # Build the tree
        self.tree = self.build_tree(X, y)
        self.feature_names = None
        
        return self
    
    def predict_single(self, x, node):
        """Predict for a single sample"""
        if node['type'] == 'leaf':
            return node['prediction']
        else:
            if x[node['feature_idx']] < node['threshold']:
                return self.predict_single(x, node['left'])
            else:
                return self.predict_single(x, node['right'])
    
    def predict(self, X):
        """Make predictions for multiple samples"""
        if hasattr(X, 'values'):
            X = X.values
        
        predictions = []
        for sample in X:
            predictions.append(self.predict_single(sample, self.tree))
        
        return np.array(predictions)
    
    def print_tree(self, node=None, depth=0, feature_names=None):
        """Print the decision tree structure"""
        if node is None:
            node = self.tree
            if feature_names is None:
                feature_names = [f"Feature_{i}" for i in range(30)]
            self.feature_names = feature_names
        
        if node['type'] == 'leaf':
            indent = "  " * depth
            print(f"{indent}└── Predict: Class {node['prediction']} (samples: {node['n_samples']}, distribution: {node['class_distribution']})")
        else:
            indent = "  " * depth
            feature_name = feature_names[node['feature_idx']] if feature_names else f"Feature_{node['feature_idx']}"
            print(f"{indent}├── If {feature_name} < {node['threshold']:.4f} (gain: {node['gain']:.4f}, samples: {node['n_samples']})")
            print(f"{indent}│   └── Then:")
            self.print_tree(node['left'], depth + 2, feature_names)
            print(f"{indent}└── Else:")
            self.print_tree(node['right'], depth + 2, feature_names)
    
    def get_feature_importance(self, feature_names=None):
        """Calculate feature importance based on gains"""
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(30)]
        
        importance = np.zeros(len(feature_names))
        
        def traverse(node, weight=1.0):
            if node['type'] == 'split':
                importance[node['feature_idx']] += node['gain'] * weight
                # Recursively traverse children with reduced weight
                left_weight = weight * (len(node['left']['class_distribution']) / node['n_samples'])
                right_weight = weight * (len(node['right']['class_distribution']) / node['n_samples'])
                traverse(node['left'], left_weight)
                traverse(node['right'], right_weight)
        
        traverse(self.tree)
        
        # Normalize
        importance = importance / np.sum(importance)
        
        # Create DataFrame
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance

# Title and description
st.title("🔬 Breast Cancer Diagnosis System")
st.markdown("### Decision Tree Implementation from Scratch")
st.markdown("---")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('data/breast-cancer.csv')
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    if 'Unnamed: 32' in df.columns:
        df = df.drop(columns=['Unnamed: 32'])
    if 'diagnosis' in df.columns:
        df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    return df

df = load_data()
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

# Sidebar - Model Parameters
st.sidebar.header("⚙️ Model Settings")
max_depth = st.sidebar.slider("Tree Depth", 1, 10, 4)
min_samples_split = st.sidebar.slider("Min Samples to Split", 2, 10, 6)
test_size = st.sidebar.slider("Test Set Size", 0.1, 0.3, 0.2)

# Main content - Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dataset", "🌳 Train Model", "📈 Results", "🎯 Predict"])

with tab1:
    st.header("Breast Cancer Dataset Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Patients", len(df))
    with col2:
        st.metric("Features", len(df.columns)-1)
    with col3:
        malignant = sum(y)
        benign = len(y) - malignant
        st.metric("Malignant Cases", f"{malignant} ({malignant/len(y)*100:.1f}%)")
    
    st.subheader("Feature Correlation Analysis")
    corr = X.corrwith(y).sort_values(ascending=False)
    
    # Smaller correlation plot
    fig, ax = plt.subplots(figsize=(8, 5))
    corr.head(15).plot(kind='barh', ax=ax, color='coral')
    ax.set_xlabel("Correlation with Malignancy")
    ax.set_title("Top 15 Most Predictive Features")
    ax.tick_params(axis='y', labelsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    
    st.subheader("Sample Data")
    st.dataframe(df.head(10))

with tab2:
    st.header("Train Decision Tree Model")
    
    if st.button("🚀 Train Model", type="primary"):
        with st.spinner("Training decision tree..."):
            # Split data
            X_train, X_test, y_train, y_test = stratified_split(X.values, y.values, test_size=test_size)
            
            # Train model
            tree = SimpleDecisionTree(
                max_depth=max_depth,
                min_samples_split=min_samples_split
            )
            tree.fit(X_train, y_train)
            
            # Store in session
            st.session_state['tree'] = tree
            st.session_state['X_test'] = X_test
            st.session_state['y_test'] = y_test
            st.session_state['X_train'] = X_train
            st.session_state['y_train'] = y_train
            
            # Evaluate
            train_pred = tree.predict(X_train)
            test_pred = tree.predict(X_test)
            
            train_acc = np.mean(train_pred == y_train)
            test_acc = np.mean(test_pred == y_test)
            
            st.session_state['train_acc'] = train_acc
            st.session_state['test_acc'] = test_acc
            st.session_state['test_pred'] = test_pred
            
            st.success("✅ Model trained successfully!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Training Accuracy", f"{train_acc:.2%}")
            with col2:
                st.metric("Test Accuracy", f"{test_acc:.2%}")
            
            # Show tree structure with proper indentation
            st.subheader("🌳 Decision Tree Structure")
            
            def tree_to_text(node, depth=0, prefix="", feature_names=X.columns):
                """Better tree visualization with proper indentation"""
                if node['type'] == 'leaf':
                    diagnosis = "Malignant" if node['prediction'] == 1 else "Benign"
                    return f"{prefix}└── {diagnosis} (n={node['n_samples']})\n"
                else:
                    result = ""
                    # Current node
                    result += f"{prefix}├── {feature_names[node['feature_idx']]} < {node['threshold']:.4f}\n"
                    # Left child
                    result += tree_to_text(node['left'], depth + 1, prefix + "│   ", feature_names)
                    # Right child
                    result += f"{prefix}└── Else:\n"
                    result += tree_to_text(node['right'], depth + 1, prefix + "    ", feature_names)
                    return result
            
            # Display in a scrollable code block with smaller font
            tree_text = tree_to_text(tree.tree, feature_names=X.columns)
            st.code(tree_text, language="text")
            
            # Optional: Add a note about tree size
            st.caption(f"📏 Tree has {count_nodes(tree.tree)} nodes (max depth: {max_depth})")

with tab3:
    st.header("Model Performance Analysis")
    
    if 'tree' in st.session_state:
        tree = st.session_state['tree']
        X_test = st.session_state['X_test']
        y_test = st.session_state['y_test']
        y_pred = st.session_state['test_pred']
        
        # Metrics
        tn = np.sum((y_test == 0) & (y_pred == 0))
        fp = np.sum((y_test == 0) & (y_pred == 1))
        fn = np.sum((y_test == 1) & (y_pred == 0))
        tp = np.sum((y_test == 1) & (y_pred == 1))
        
        # Metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{(tp+tn)/(tp+tn+fp+fn):.2%}")
        with col2:
            precision = tp/(tp+fp) if tp+fp>0 else 0
            st.metric("Precision", f"{precision:.2%}")
        with col3:
            recall = tp/(tp+fn) if tp+fn>0 else 0
            st.metric("Recall", f"{recall:.2%}")
        with col4:
            f1 = 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn)>0 else 0
            st.metric("F1 Score", f"{f1:.2%}")
        
        # Confusion Matrix - SMALLER
        st.subheader("Confusion Matrix")
        cm = np.array([[tn, fp], [fn, tp]])
        
        # Create compact figure
        fig, ax = plt.subplots(figsize=(3.8, 3.0))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Benign', 'Malignant'],
                    yticklabels=['Benign', 'Malignant'],
                    ax=ax,
                annot_kws={'size': 10})
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("Actual", fontsize=9)
        ax.set_title("Confusion Matrix", fontsize=10)
        ax.tick_params(axis='both', labelsize=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)
        
        # Clinical Interpretation
        st.subheader("📋 Clinical Interpretation")
        col1, col2 = st.columns(2)
        with col1:
            if fn > 0:
                st.error(f"⚠️ **{fn} False Negatives** - Missed cancer cases")
            else:
                st.success("✅ **No False Negatives** - All cancers detected")
        with col2:
            if fp > 0:
                st.warning(f"📌 **{fp} False Positives** - Unnecessary follow-ups")
            else:
                st.success("✅ **No False Positives** - No unnecessary procedures")

        # Feature Importance Table
        st.subheader("Feature Importance")
        importance = tree.get_feature_importance(feature_names=X.columns.tolist())
        importance = importance[importance['importance'] > 0.000001].reset_index(drop=True)
        st.dataframe(importance, use_container_width=True, hide_index=True)
        
    else:
        st.info("Train a model first in the 'Train Model' tab")

with tab4:
    st.header("Make a Prediction")
    
    if 'tree' in st.session_state:
        tree = st.session_state['tree']
        feature_names = X.columns.tolist()
        
        st.markdown("Enter patient data for diagnosis:")
        
        # Create input fields
        input_data = {}
        cols = st.columns(3)
        
        # Use top 10 most important features for simplicity
        importance = tree.get_feature_importance(feature_names=feature_names)
        top_features = importance.head(10)['feature'].tolist()
        
        for i, feature in enumerate(top_features):
            with cols[i % 3]:
                median_val = float(X[feature].median())
                input_data[feature] = st.number_input(
                    feature,
                    value=median_val,
                    format="%.4f",
                    key=feature
                )
        
        if st.button("🔍 Diagnose", type="primary"):
            # Create array with all features (using defaults for non-top features)
            all_features = []
            for f in feature_names:
                if f in input_data:
                    all_features.append(input_data[f])
                else:
                    all_features.append(float(X[f].median()))
            
            prediction = tree.predict(np.array(all_features).reshape(1, -1))[0]
            
            st.markdown("---")
            if prediction == 1:
                st.error("### ⚠️ Diagnosis: **MALIGNANT**")
                st.warning("The model predicts cancerous tumor. Immediate medical consultation recommended.")
            else:
                st.success("### ✅ Diagnosis: **BENIGN**")
                st.info("The model predicts non-cancerous tumor. Regular check-ups recommended.")
    else:
        st.info("Please train a model first in the 'Train Model' tab")

st.markdown("---")
st.markdown("📝 **Note:** This is a decision tree implemented from scratch for educational purposes.")