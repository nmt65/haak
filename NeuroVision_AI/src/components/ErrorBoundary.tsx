import { Component, type ErrorInfo, type ReactNode } from "react";

interface ErrorBoundaryProps {
  children: ReactNode;
  areaName: string;
}

interface ErrorBoundaryState {
  error: Error | null;
}

/**
 * Catches rendering errors inside one area of the app so the rest keeps working.
 * React only supports this as a class component.
 */
export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error(`Eroare de randare în „${this.props.areaName}”:`, error, info.componentStack);
  }

  private reset = () => this.setState({ error: null });

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div role="alert" className="paper-well p-5 text-sm">
        <p className="font-bold text-live">Secțiunea „{this.props.areaName}” nu s-a putut afișa.</p>
        <p className="mt-1 text-graphite">{this.state.error.message}</p>
        <button type="button" onClick={this.reset} className="mt-3 rounded-lg border border-rule bg-paper px-3 py-1.5 font-semibold text-ink hover:border-ink">
          Reîncarcă secțiunea
        </button>
      </div>
    );
  }
}
