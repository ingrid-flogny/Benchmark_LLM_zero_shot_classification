import pandas as pd
import pandas as pd
import re
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_fscore_support
import plotly.express as px

from functions import extract_class_from_result


class Benchmark_LLM:

    def __init__(self, df : pd.DataFrame, true_label_colname : str, categories : list, dict_model_results : dict):
        self.df = df
        self.labelisation = true_label_colname
        self.categories = categories
        self.model_name_results = dict_model_results

    def extract_LLM_label_from_answers(self):
        """
        :return:
        """
        for model in [*self.model_name_results]:
            self.df[f'CLASSIFICATION_{model}'] = self.df[f'{self.model_name_results[model]}'].apply(
                lambda x: extract_class_from_result(x, self.categories))

    def confusion_matrix_one_model(self, model_name : str) -> pd.DataFrame:
        conf_matrix = confusion_matrix(self.df[self.labelisation], self.df[f'CLASSIFICATION_{model_name}'],
                                       labels=self.categories + ['INEXPLOITABLE'])
        conf_matrix_df = pd.DataFrame(conf_matrix, index=self.categories + ['INEXPLOITABLE'],
                                            columns=self.categories + ['INEXPLOITABLE'])
        return conf_matrix_df


    def metrics_matrix(self):
        result_matrix = pd.DataFrame(columns=['Model', 'Category', 'Metrics', 'Value'])
        for model in [*self.model_name_results]:
            precision, recall, f1_score, _ = precision_recall_fscore_support(self.df[self.labelisation],
                                                                             self.df[f'CLASSIFICATION_{model}'],
                                                                             labels=self.categories,
                                                                             average=None)
            dict_metrics = {'precision': precision, 'recall': recall, 'f1-score': f1_score}
            for metric in [*dict_metrics]:
                temp_df = pd.DataFrame(
                    data={'Model': model, 'Category': self.categories, 'Metrics': metric, 'Value': dict_metrics[metric]})
                result_matrix = pd.concat([result_matrix, temp_df])
        self.result_matrix = result_matrix

        return result_matrix

    def plot_linear_polar(self, metric : str):
        result_matrix_df = self.result_matrix[self.result_matrix['Metrics'] == metric]
        fig = px.line_polar(result_matrix_df, r='Value', theta='Category', color='Model', line_close=True,
                            color_discrete_sequence=px.colors.sequential.Plasma_r)
        return fig
