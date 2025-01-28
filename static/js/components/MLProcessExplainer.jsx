import React, { useState } from 'react';
import { Camera, FileText, Database, RefreshCw, Award, BarChart2, AlertCircle, ChevronDown, ChevronUp } from 'lucide-react';

const MLProcessExplainer = () => {
    const [activeStep, setActiveStep] = useState(null);

    const steps = [
        {
            id: 'data-upload',
            title: 'Data Upload',
            Icon: FileText,
            description: 'Upload your data in CSV format',
            details: [
                'Your data is securely uploaded',
                'We automatically detect data types',
                'We check for data quality issues'
            ],
            friendlyExplanation: "Like organizing a spreadsheet - we make sure everything is in the right place."
        },
        {
            id: 'preprocessing',
            title: 'Data Cleaning',
            Icon: RefreshCw,
            description: 'We clean and prepare your data',
            details: [
                'Missing values are handled automatically',
                'Text data is converted to numbers',
                'Data is scaled for better results'
            ],
            friendlyExplanation: "We organize and clean your data so the AI can understand it better."
        },
        {
            id: 'analysis',
            title: 'Task Analysis',
            Icon: Database,
            description: 'We determine what type of problem to solve',
            details: [
                'Detect if predicting numbers or categories',
                'Identify target column',
                'Analyze important features'
            ],
            friendlyExplanation: "We figure out what you're trying to predict and what information is most helpful."
        },
        {
            id: 'model-training',
            title: 'Model Training',
            Icon: BarChart2,
            description: 'AI models learn from your data',
            details: [
                'Multiple AI models are tested',
                'Each model is fine-tuned',
                'Best performing model is selected'
            ],
            friendlyExplanation: "Like having multiple experts analyze your data - we pick the one that works best."
        },
        {
            id: 'evaluation',
            title: 'Results & Insights',
            Icon: Award,
            description: 'Get clear results and insights',
            details: [
                'See model performance',
                'Get plain-English explanations',
                'Receive improvement suggestions'
            ],
            friendlyExplanation: "We show you how well the AI performed and what the results mean."
        }
    ];

    const handleStepClick = (stepId) => {
        setActiveStep(activeStep === stepId ? null : stepId);
    };

    return (
        <div className="max-w-4xl mx-auto p-4">
            <div className="mb-8">
                <h2 className="text-2xl font-bold mb-2">How Our AutoML Platform Works</h2>
                <p className="text-gray-600">
                    We make machine learning easy by automating the complex parts while keeping you informed.
                </p>
            </div>

            <div className="space-y-4">
                {steps.map((step) => (
                    <div
                        key={step.id}
                        className={`bg-white rounded-lg shadow-md p-4 cursor-pointer 
                            ${activeStep === step.id ? 'ring-2 ring-blue-500' : ''}`}
                    >
                        <div 
                            className="flex items-center justify-between"
                            onClick={() => handleStepClick(step.id)}
                        >
                            <div className="flex items-center space-x-4">
                                <div className={`p-2 rounded-full 
                                    ${activeStep === step.id ? 'bg-blue-100 text-blue-600' : 'bg-gray-100'}`}
                                >
                                    <step.Icon size={24} />
                                </div>
                                <div>
                                    <h3 className="font-semibold text-lg">{step.title}</h3>
                                    <p className="text-gray-600">{step.description}</p>
                                </div>
                            </div>
                            {activeStep === step.id ? 
                                <ChevronUp className="text-gray-400" size={20} /> : 
                                <ChevronDown className="text-gray-400" size={20} />
                            }
                        </div>

                        {activeStep === step.id && (
                            <div className="mt-4 pl-14">
                                <div className="bg-gray-50 p-4 rounded-lg">
                                    <h4 className="font-medium mb-2">What happens in this step:</h4>
                                    <ul className="list-disc pl-5 space-y-2 mb-4">
                                        {step.details.map((detail, i) => (
                                            <li key={i} className="text-gray-600">{detail}</li>
                                        ))}
                                    </ul>
                                    <div className="bg-blue-50 p-4 rounded-lg">
                                        <div className="flex items-start space-x-2">
                                            <AlertCircle className="text-blue-500 mt-1" size={20} />
                                            <div>
                                                <h5 className="font-medium text-blue-700">In simple terms:</h5>
                                                <p className="text-blue-600">{step.friendlyExplanation}</p>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                ))}
            </div>

            <div className="mt-8 bg-green-50 p-4 rounded-lg">
                <h3 className="font-semibold text-green-700 mb-2">Why use our platform?</h3>
                <ul className="space-y-2 text-green-600">
                    <li>• No coding required - everything is automated</li>
                    <li>• Clear explanations at every step</li>
                    <li>• Professional-grade AI results without technical expertise</li>
                    <li>• Understand your data better with detailed insights</li>
                </ul>
            </div>
        </div>
    );
};

export default MLProcessExplainer;