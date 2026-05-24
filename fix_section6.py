import json

notebook_path = 'loan_approval_prediction.ipynb'
cell_id = 'e5bf7d25'
new_source = [
    'print("Classification reports for all models")\n',
    'for model_name, predictions in model_predictions.items():\n',
    '    print("\\n" + "=" * 80)\n',
    '    print(f"Classification report for {model_name}")\n',
    '    print(classification_report(y_test, predictions, target_names=["Not Approved", "Approved"], zero_division=0))\n',
    '\n',
    'print("\\nConfusion matrix heatmaps")\n',
    'fig, axes = plt.subplots(2, 3, figsize=(18, 10))\n',
    'axes = axes.flatten()\n',
    'for index, (model_name, predictions) in enumerate(model_predictions.items()):\n',
    '    cm = confusion_matrix(y_test, predictions)\n',
    '    sns.heatmap(\n',
    '        cm,\n',
    '        annot=True,\n',
    '        fmt="d",\n',
    '        cmap="Blues",\n',
    '        cbar=False,\n',
    '        ax=axes[index],\n',
    '        xticklabels=["Pred N", "Pred Y"],\n',
    '        yticklabels=["True N", "True Y"],\n',
    '    )\n',
    '    axes[index].set_title(model_name)\n',
    '    axes[index].set_xlabel("Predicted Label")\n',
    '    axes[index].set_ylabel("True Label")\n',
    'plt.tight_layout()\n',
    'plt.show()\n',
    '\n',
    'comparison_df = pd.DataFrame(model_results).T.reset_index().rename(columns={"index": "Model"})\n',
    'comparison_df = comparison_df.sort_values("F1-Score", ascending=False).reset_index(drop=True)\n',
    '\n',
    'print("Comparison DataFrame")\n',
    'display(comparison_df)\n',
    '\n',
    'print("Grouped bar chart of model metrics")\n',
    'comparison_melted = comparison_df.melt(id_vars="Model", var_name="Metric", value_name="Score")\n',
    'plt.figure(figsize=(14, 7))\n',
    'ax = sns.barplot(data=comparison_melted, x="Model", y="Score", hue="Metric", palette="tab10")\n',
    'ax.set_title("Model Comparison Across Metrics")\n',
    'ax.set_xlabel("Model")\n',
    'ax.set_ylabel("Score")\n',
    'plt.xticks(rotation=20, ha="right")\n',
    'plt.ylim(0, 1.05)\n',
    'plt.tight_layout()\n',
    'plt.show()\n',
    '\n',
    'best_model_name = comparison_df.iloc[0]["Model"]\n',
    'best_model = trained_models[best_model_name]\n',
    'best_model_metrics = comparison_df.iloc[0].to_dict()\n',
    '\n',
    'print(f"BEST MODEL by F1-score: {best_model_name}")\n',
    'print("Best model metrics:")\n',
    'print(best_model_metrics)\n',
    '\n',
    'cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring="f1_weighted")\n',
    'print(f"5-fold CV for {best_model_name}")\n',
    'print(f"Mean F1-score: {cv_scores.mean():.4f}")\n',
    'print(f"Std F1-score:  {cv_scores.std():.4f}")\n',
    'print(f"Mean ± Std:    {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")\n',
]

with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

replaced = False
for cell in notebook.get('cells', []):
    if cell.get('id') == cell_id:
        cell['source'] = new_source
        replaced = True
        break

if not replaced:
    raise SystemExit(f'Cell {cell_id} not found')

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print('Section 6 rewritten successfully')
