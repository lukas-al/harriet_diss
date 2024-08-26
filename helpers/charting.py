""" Somewhere to put charts """
import seaborn as sns
import matplotlib.pyplot as plt

def rmse_plot(
    evaluation_score_list,
    evaluation_dates,
    title,
):
    
    fig, ax = plt.subplots(figsize=(16, 8))
    fig = sns.lineplot(x=evaluation_dates, y=evaluation_score_list, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("RMSE")
    
    return fig


def hedgehog_plot(
    model_results,
    target_series,
    title = 'Model Nowcasts',
    log_axis=False,
):
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    fig = sns.lineplot(target_series, ax=ax)
    
    for idx, (scores, preds) in enumerate(zip(model_results['scores'], model_results['predictions'])): 
        if idx==5:
            sns.lineplot(preds, color='red',ax=ax, alpha=1)
            print(preds)
        
        sns.lineplot(preds, color='grey',ax=ax, alpha=0.33)
        
    ax.set_xlabel("Date")
    ax.set_ylabel('Quarterly GDP growth')
    ax.set_title(title)
    
    if log_axis:
        ax.set_yscale('symlog')
    
    return fig