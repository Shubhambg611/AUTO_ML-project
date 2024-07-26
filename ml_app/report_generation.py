import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

def generate_report(results, insights):
    # Generate a report with results and insights
    report = {
        'results': results,
        'insights': insights
    }
    return report
