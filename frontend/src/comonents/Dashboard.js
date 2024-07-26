import React, { useState } from 'react';
import FileUpload from './FileUpload';
import Results from './Results';

function Dashboard() {
    const [results, setResults] = useState(null);

    const handleUploadSuccess = (data) => {
        setResults(data);
    };

    return (
        <div>
            <h1>Automated Machine Learning Model Selector</h1>
            <FileUpload onUploadSuccess={handleUploadSuccess} />
            {results && <Results data={results} />}
        </div>
    );
}

export default Dashboard;
