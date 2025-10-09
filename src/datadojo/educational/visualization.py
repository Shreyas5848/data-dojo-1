"""Progress visualization for DataDojo.

Provides visual representations of learning progress, skill assessments,
and concept mastery using matplotlib and plotly.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import json
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from ..models.progress_tracker import ProgressTracker


class ProgressVisualizer:
    """Visualize learning progress and achievements."""

    def __init__(self, use_plotly: bool = False):
        """Initialize the progress visualizer.

        Args:
            use_plotly: If True, use Plotly for interactive charts.
                       If False, use Matplotlib for static charts.
        """
        self.use_plotly = use_plotly and HAS_PLOTLY

        if not self.use_plotly and not HAS_MATPLOTLIB:
            raise ImportError(
                "Neither matplotlib nor plotly is installed. "
                "Install with: pip install matplotlib or pip install plotly"
            )

    def plot_progress_timeline(
        self,
        progress: ProgressTracker,
        output_path: Optional[str] = None
    ) -> Optional[Figure]:
        """Plot progress over time showing completed steps.

        Args:
            progress: ProgressTracker with progress history
            output_path: Optional path to save the figure

        Returns:
            Matplotlib Figure if using matplotlib, None if using plotly
        """
        if not progress.completed_steps:
            print("No completed steps to visualize")
            return None

        # Extract timeline data
        dates = []
        cumulative_count = []

        # Sort by completion date
        steps_with_dates = [
            (step_id, progress.step_completion_dates.get(step_id))
            for step_id in progress.completed_steps
            if step_id in progress.step_completion_dates
        ]
        steps_with_dates.sort(key=lambda x: x[1])

        for i, (step_id, date_str) in enumerate(steps_with_dates, 1):
            dates.append(datetime.fromisoformat(date_str))
            cumulative_count.append(i)

        if self.use_plotly:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=cumulative_count,
                mode='lines+markers',
                name='Completed Steps',
                line=dict(color='#2E86AB', width=3),
                marker=dict(size=8)
            ))

            fig.update_layout(
                title='Learning Progress Timeline',
                xaxis_title='Date',
                yaxis_title='Cumulative Steps Completed',
                template='plotly_white',
                hovermode='x unified'
            )

            if output_path:
                fig.write_html(output_path)
            else:
                fig.show()
            return None
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(dates, cumulative_count, marker='o', linewidth=2,
                   markersize=6, color='#2E86AB')

            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Cumulative Steps Completed', fontsize=12)
            ax.set_title('Learning Progress Timeline', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Format x-axis dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            fig.autofmt_xdate()

            plt.tight_layout()

            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close(fig)

            return fig

    def plot_skill_radar(
        self,
        progress: ProgressTracker,
        output_path: Optional[str] = None
    ) -> Optional[Figure]:
        """Plot radar chart of skill assessments.

        Args:
            progress: ProgressTracker with skill assessments
            output_path: Optional path to save the figure

        Returns:
            Matplotlib Figure if using matplotlib, None if using plotly
        """
        if not progress.skill_assessments:
            print("No skill assessments to visualize")
            return None

        skills = list(progress.skill_assessments.keys())
        scores = list(progress.skill_assessments.values())

        if self.use_plotly:
            fig = go.Figure()

            fig.add_trace(go.Scatterpolar(
                r=scores,
                theta=skills,
                fill='toself',
                name='Skill Level',
                fillcolor='rgba(46, 134, 171, 0.5)',
                line=dict(color='#2E86AB', width=2)
            ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )
                ),
                title='Skill Assessment Profile',
                showlegend=False,
                template='plotly_white'
            )

            if output_path:
                fig.write_html(output_path)
            else:
                fig.show()
            return None
        else:
            import numpy as np

            # Number of variables
            num_vars = len(skills)

            # Compute angle for each axis
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

            # Complete the circle
            scores_plot = scores + [scores[0]]
            angles_plot = angles + [angles[0]]

            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
            ax.plot(angles_plot, scores_plot, 'o-', linewidth=2, color='#2E86AB')
            ax.fill(angles_plot, scores_plot, alpha=0.25, color='#2E86AB')

            # Fix axis to go in the right order
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)

            # Set labels
            ax.set_xticks(angles)
            ax.set_xticklabels(skills)

            # Set y-axis limits
            ax.set_ylim(0, 100)
            ax.set_yticks([20, 40, 60, 80, 100])
            ax.set_yticklabels(['20', '40', '60', '80', '100'])

            ax.set_title('Skill Assessment Profile', fontsize=14, fontweight='bold', pad=20)
            ax.grid(True)

            plt.tight_layout()

            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close(fig)

            return fig

    def plot_concept_mastery(
        self,
        progress: ProgressTracker,
        all_concepts: List[str],
        output_path: Optional[str] = None
    ) -> Optional[Figure]:
        """Plot bar chart of concept learning progress.

        Args:
            progress: ProgressTracker with learned concepts
            all_concepts: List of all available concepts
            output_path: Optional path to save the figure

        Returns:
            Matplotlib Figure if using matplotlib, None if using plotly
        """
        if not all_concepts:
            print("No concepts to visualize")
            return None

        # Calculate mastery for each concept
        concept_status = []
        for concept in all_concepts:
            if concept in progress.learned_concepts:
                concept_status.append(('Learned', concept))
            else:
                concept_status.append(('Not Started', concept))

        learned = [c for s, c in concept_status if s == 'Learned']
        not_started = [c for s, c in concept_status if s == 'Not Started']

        if self.use_plotly:
            fig = go.Figure()

            if learned:
                fig.add_trace(go.Bar(
                    y=learned,
                    x=[100] * len(learned),
                    orientation='h',
                    name='Learned',
                    marker=dict(color='#06A77D')
                ))

            if not_started:
                fig.add_trace(go.Bar(
                    y=not_started,
                    x=[0] * len(not_started),
                    orientation='h',
                    name='Not Started',
                    marker=dict(color='#D5D5D5')
                ))

            fig.update_layout(
                title='Concept Mastery Progress',
                xaxis_title='Mastery (%)',
                yaxis_title='Concept',
                barmode='overlay',
                template='plotly_white',
                height=max(400, len(all_concepts) * 30)
            )

            if output_path:
                fig.write_html(output_path)
            else:
                fig.show()
            return None
        else:
            fig, ax = plt.subplots(figsize=(10, max(6, len(all_concepts) * 0.4)))

            y_positions = range(len(all_concepts))
            colors = ['#06A77D' if c in learned else '#D5D5D5' for c in all_concepts]
            values = [100 if c in learned else 0 for c in all_concepts]

            bars = ax.barh(y_positions, values, color=colors)

            ax.set_yticks(y_positions)
            ax.set_yticklabels(all_concepts)
            ax.set_xlabel('Mastery (%)', fontsize=12)
            ax.set_title('Concept Mastery Progress', fontsize=14, fontweight='bold')
            ax.set_xlim(0, 100)

            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#06A77D', label='Learned'),
                Patch(facecolor='#D5D5D5', label='Not Started')
            ]
            ax.legend(handles=legend_elements, loc='lower right')

            plt.tight_layout()

            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close(fig)

            return fig

    def plot_completion_percentage(
        self,
        progress: ProgressTracker,
        total_steps: int,
        output_path: Optional[str] = None
    ) -> Optional[Figure]:
        """Plot completion percentage gauge.

        Args:
            progress: ProgressTracker instance
            total_steps: Total number of steps in the project
            output_path: Optional path to save the figure

        Returns:
            Matplotlib Figure if using matplotlib, None if using plotly
        """
        completion = progress.get_completion_percentage(total_steps)

        if self.use_plotly:
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=completion,
                title={'text': "Project Completion"},
                delta={'reference': 0},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#2E86AB"},
                    'steps': [
                        {'range': [0, 33], 'color': "#FFE5E5"},
                        {'range': [33, 66], 'color': "#FFF4E5"},
                        {'range': [66, 100], 'color': "#E5F7F0"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 100
                    }
                }
            ))

            fig.update_layout(
                template='plotly_white',
                height=400
            )

            if output_path:
                fig.write_html(output_path)
            else:
                fig.show()
            return None
        else:
            fig, ax = plt.subplots(figsize=(8, 6))

            # Create a simple bar chart for completion
            ax.barh(['Completion'], [completion], color='#2E86AB', height=0.5)
            ax.barh(['Completion'], [100 - completion], left=[completion],
                   color='#D5D5D5', height=0.5)

            ax.set_xlim(0, 100)
            ax.set_xlabel('Percentage (%)', fontsize=12)
            ax.set_title(f'Project Completion: {completion:.1f}%',
                        fontsize=14, fontweight='bold')

            # Add percentage text
            ax.text(completion / 2, 0, f'{completion:.1f}%',
                   ha='center', va='center', fontsize=16, fontweight='bold',
                   color='white')

            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)

            plt.tight_layout()

            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close(fig)

            return fig

    def generate_progress_dashboard(
        self,
        progress: ProgressTracker,
        total_steps: int,
        all_concepts: List[str],
        output_dir: str
    ) -> None:
        """Generate a complete progress dashboard with all visualizations.

        Args:
            progress: ProgressTracker instance
            total_steps: Total number of steps
            all_concepts: List of all available concepts
            output_dir: Directory to save visualizations
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        ext = '.html' if self.use_plotly else '.png'

        print(f"Generating progress dashboard in {output_dir}...")

        # Generate all visualizations
        self.plot_completion_percentage(
            progress,
            total_steps,
            str(output_path / f'completion{ext}')
        )
        print(f"✓ Completion gauge saved")

        if progress.completed_steps:
            self.plot_progress_timeline(
                progress,
                str(output_path / f'timeline{ext}')
            )
            print(f"✓ Progress timeline saved")

        if progress.skill_assessments:
            self.plot_skill_radar(
                progress,
                str(output_path / f'skills{ext}')
            )
            print(f"✓ Skill radar chart saved")

        if all_concepts:
            self.plot_concept_mastery(
                progress,
                all_concepts,
                str(output_path / f'concepts{ext}')
            )
            print(f"✓ Concept mastery chart saved")

        print(f"\n Dashboard generated successfully!")
        print(f"View your progress visualizations in: {output_dir}")


def create_visualizer(use_plotly: bool = False) -> ProgressVisualizer:
    """Factory function to create a ProgressVisualizer.

    Args:
        use_plotly: If True, use Plotly for interactive charts

    Returns:
        ProgressVisualizer instance
    """
    return ProgressVisualizer(use_plotly=use_plotly)
