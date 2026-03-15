import { AlertTriangle } from 'lucide-react';

const Disclaimer = () => {
    return (
        <div className="disclaimer-alert">
            <AlertTriangle size={24} className="icon" />
            <div className="content">
                <strong>Protoype Screening Tool</strong>
                <p>This application is for demonstration and research purposes only. It is not a medical device and does not provide a diagnosis. If you suspect a stroke, call emergency services immediately.</p>
            </div>

            <style>{`
                .disclaimer-alert {
                    display: flex;
                    gap: var(--space-md);
                    background: hsl(var(--status-warning), 0.1);
                    border: 1px solid hsl(var(--status-warning), 0.3);
                    padding: var(--space-md);
                    border-radius: var(--radius-md);
                    margin-top: var(--space-xl);
                    font-size: 0.85rem;
                    color: hsl(var(--text-main));
                }
                .icon { color: hsl(var(--status-warning)); flex-shrink: 0; }
                .content strong { display: block; margin-bottom: 4px; color: hsl(var(--text-main)); }
                .content p { margin: 0; color: hsl(var(--text-muted)); line-height: 1.4; }
            `}</style>
        </div>
    );
};

export default Disclaimer;
