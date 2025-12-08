"""
Learning Projects Interface for DataDojo Web Dashboard
Displays built-in projects and allows users to start guided learning experiences
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
from pathlib import Path
import os

# Import domain projects
from ..domains import get_registry
from ..models.learning_project import Domain, Difficulty
from .project_notebook_generator import generate_project_notebook


def get_project_icon(domain: Domain) -> str:
    """Get icon for domain."""
    icons = {
        Domain.ECOMMERCE: "🛒",
        Domain.HEALTHCARE: "🏥",
        Domain.FINANCE: "💰",
    }
    return icons.get(domain, "📁")


def get_difficulty_badge(difficulty: Difficulty) -> str:
    """Get HTML badge for difficulty level."""
    colors = {
        Difficulty.BEGINNER: ("#22C55E", "Beginner"),
        Difficulty.INTERMEDIATE: ("#F59E0B", "Intermediate"),
        Difficulty.ADVANCED: ("#EF4444", "Advanced"),
    }
    color, label = colors.get(difficulty, ("#94A3B8", "Unknown"))
    return f"""<span style="
        background: {color}; 
        color: white; 
        padding: 0.25rem 0.75rem; 
        border-radius: 9999px; 
        font-size: 0.75rem; 
        font-weight: 600;
        text-transform: uppercase;
    ">{label}</span>"""


def get_domain_color(domain: Domain) -> str:
    """Get color for domain."""
    colors = {
        Domain.ECOMMERCE: "#3B82F6",
        Domain.HEALTHCARE: "#14B8A6",
        Domain.FINANCE: "#8B5CF6",
    }
    return colors.get(domain, "#64748B")


def load_user_projects() -> Dict[str, Any]:
    """Load user's project progress."""
    progress_file = Path.home() / ".datadojo" / "project_progress.json"
    if progress_file.exists():
        try:
            with open(progress_file, 'r') as f:
                return json.load(f)
        except:
            pass
    return {"started_projects": {}, "completed_projects": []}


def save_user_projects(data: Dict[str, Any]) -> None:
    """Save user's project progress."""
    progress_file = Path.home() / ".datadojo" / "project_progress.json"
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    with open(progress_file, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def render_learning_projects():
    """Render the Learning Projects page."""
    
    # Page Header
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 2rem;
        text-align: center;
    ">
        <h1 style="color: #F8FAFC; font-size: 2rem; margin: 0 0 0.5rem 0; font-weight: 700;">
            🎓 Learning Projects
        </h1>
        <p style="color: #94A3B8; font-size: 1rem; margin: 0;">
            Hands-on projects to master data science skills across multiple domains
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load registry and projects
    registry = get_registry()
    all_projects = registry.get_all_projects()
    user_progress = load_user_projects()
    
    # Filters
    st.markdown("""
    <h3 style="color: #F8FAFC; margin: 1rem 0; font-size: 1.125rem; font-weight: 600;">
        Filter Projects
    </h3>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        domain_filter = st.selectbox(
            "Domain",
            options=["All", "E-Commerce", "Healthcare", "Finance"],
            key="project_domain_filter"
        )
    
    with col2:
        difficulty_filter = st.selectbox(
            "Difficulty",
            options=["All", "Beginner", "Intermediate", "Advanced"],
            key="project_difficulty_filter"
        )
    
    with col3:
        status_filter = st.selectbox(
            "Status",
            options=["All", "Not Started", "In Progress", "Completed"],
            key="project_status_filter"
        )
    
    # Apply filters
    filtered_projects = all_projects
    
    if domain_filter != "All":
        domain_map = {"E-Commerce": Domain.ECOMMERCE, "Healthcare": Domain.HEALTHCARE, "Finance": Domain.FINANCE}
        filtered_projects = [p for p in filtered_projects if p.domain == domain_map.get(domain_filter)]
    
    if difficulty_filter != "All":
        diff_map = {"Beginner": Difficulty.BEGINNER, "Intermediate": Difficulty.INTERMEDIATE, "Advanced": Difficulty.ADVANCED}
        filtered_projects = [p for p in filtered_projects if p.difficulty == diff_map.get(difficulty_filter)]
    
    if status_filter != "All":
        if status_filter == "Not Started":
            filtered_projects = [p for p in filtered_projects if p.id not in user_progress.get("started_projects", {})]
        elif status_filter == "In Progress":
            filtered_projects = [p for p in filtered_projects 
                               if p.id in user_progress.get("started_projects", {}) 
                               and p.id not in user_progress.get("completed_projects", [])]
        elif status_filter == "Completed":
            filtered_projects = [p for p in filtered_projects if p.id in user_progress.get("completed_projects", [])]
    
    # Stats row
    total = len(all_projects)
    started = len(user_progress.get("started_projects", {}))
    completed = len(user_progress.get("completed_projects", []))
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div style="background: #1E293B; padding: 1rem; border-radius: 8px; border: 1px solid #334155; text-align: center;">
            <div style="color: #3B82F6; font-size: 1.5rem; font-weight: 700;">{total}</div>
            <div style="color: #94A3B8; font-size: 0.8rem;">Total Projects</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background: #1E293B; padding: 1rem; border-radius: 8px; border: 1px solid #334155; text-align: center;">
            <div style="color: #F59E0B; font-size: 1.5rem; font-weight: 700;">{started}</div>
            <div style="color: #94A3B8; font-size: 0.8rem;">In Progress</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="background: #1E293B; padding: 1rem; border-radius: 8px; border: 1px solid #334155; text-align: center;">
            <div style="color: #22C55E; font-size: 1.5rem; font-weight: 700;">{completed}</div>
            <div style="color: #94A3B8; font-size: 0.8rem;">Completed</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        completion_pct = int((completed / total * 100)) if total > 0 else 0
        st.markdown(f"""
        <div style="background: #1E293B; padding: 1rem; border-radius: 8px; border: 1px solid #334155; text-align: center;">
            <div style="color: #14B8A6; font-size: 1.5rem; font-weight: 700;">{completion_pct}%</div>
            <div style="color: #94A3B8; font-size: 0.8rem;">Completion</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Display projects
    if not filtered_projects:
        st.info("No projects match your filters. Try adjusting the filters above.")
        return
    
    # Group by domain
    projects_by_domain = {}
    for project in filtered_projects:
        domain_name = project.domain.value
        if domain_name not in projects_by_domain:
            projects_by_domain[domain_name] = []
        projects_by_domain[domain_name].append(project)
    
    # Render each domain section
    for domain_name, projects in projects_by_domain.items():
        domain_enum = projects[0].domain
        domain_color = get_domain_color(domain_enum)
        domain_icon = get_project_icon(domain_enum)
        
        st.markdown(f"""
        <h2 style="
            color: #F8FAFC; 
            margin: 1.5rem 0 1rem 0; 
            font-size: 1.25rem; 
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        ">
            <span style="font-size: 1.5rem;">{domain_icon}</span>
            {domain_name.replace('_', ' ').title()} Projects
            <span style="
                background: {domain_color}20;
                color: {domain_color};
                padding: 0.25rem 0.5rem;
                border-radius: 4px;
                font-size: 0.75rem;
                margin-left: 0.5rem;
            ">{len(projects)} projects</span>
        </h2>
        """, unsafe_allow_html=True)
        
        # Sort by difficulty
        difficulty_order = {Difficulty.BEGINNER: 0, Difficulty.INTERMEDIATE: 1, Difficulty.ADVANCED: 2}
        projects_sorted = sorted(projects, key=lambda p: difficulty_order.get(p.difficulty, 99))
        
        for project in projects_sorted:
            render_project_card(project, user_progress, domain_color)


def render_project_card(project, user_progress: Dict, domain_color: str):
    """Render a single project card."""
    
    # Determine status
    is_started = project.id in user_progress.get("started_projects", {})
    is_completed = project.id in user_progress.get("completed_projects", [])
    
    if is_completed:
        status_badge = '<span style="background: #22C55E; color: white; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.7rem;">✓ COMPLETED</span>'
        border_color = "#22C55E"
    elif is_started:
        status_badge = '<span style="background: #F59E0B; color: white; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.7rem;">IN PROGRESS</span>'
        border_color = "#F59E0B"
    else:
        status_badge = ""
        border_color = "#334155"
    
    # Card container
    with st.container():
        st.markdown(f"""
        <div style="
            background: #1E293B;
            border: 1px solid {border_color};
            border-radius: 12px;
            padding: 1.25rem;
            margin-bottom: 1rem;
            transition: all 0.2s ease;
        ">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.75rem;">
                <div>
                    <h3 style="color: #F8FAFC; margin: 0 0 0.5rem 0; font-size: 1.1rem; font-weight: 600;">
                        {project.name}
                    </h3>
                    {get_difficulty_badge(project.difficulty)} {status_badge}
                </div>
            </div>
            <p style="color: #CBD5E1; font-size: 0.9rem; margin: 0.75rem 0; line-height: 1.5;">
                {project.description}
            </p>
            <div style="color: #94A3B8; font-size: 0.8rem; margin-top: 0.75rem;">
                <strong style="color: #F8FAFC;">Learning Objectives:</strong>
            </div>
            <ul style="color: #94A3B8; font-size: 0.8rem; margin: 0.5rem 0; padding-left: 1.25rem;">
                {''.join(f'<li style="margin: 0.25rem 0;">{obj}</li>' for obj in project.expected_outcomes[:4])}
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Action buttons
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if is_completed:
                if st.button("🔄 Restart", key=f"restart_{project.id}", type="secondary"):
                    restart_project(project.id, user_progress)
                    st.rerun()
            elif is_started:
                if st.button("▶️ Continue", key=f"continue_{project.id}", type="primary"):
                    st.session_state.active_project = project.id
                    st.session_state.show_project_detail = True
                    st.rerun()
            else:
                if st.button("🚀 Start Project", key=f"start_{project.id}", type="primary"):
                    start_project(project.id, user_progress)
                    st.session_state.active_project = project.id
                    st.session_state.show_project_detail = True
                    st.rerun()
        
        with col2:
            if st.button("📋 Details", key=f"details_{project.id}"):
                st.session_state.active_project = project.id
                st.session_state.show_project_detail = True
                st.rerun()
        
        with col3:
            if is_started and not is_completed:
                if st.button("✅ Mark Complete", key=f"complete_{project.id}"):
                    complete_project(project.id, user_progress)
                    st.rerun()


def start_project(project_id: str, user_progress: Dict):
    """Start a project and generate its notebook."""
    if "started_projects" not in user_progress:
        user_progress["started_projects"] = {}
    
    # Generate notebook for the project
    registry = get_registry()
    project = registry.get_project_by_id(project_id)
    notebook_path = None
    
    if project:
        try:
            notebook_path = generate_project_notebook(project)
            st.success(f"📓 Generated learning notebook: {notebook_path}")
        except Exception as e:
            st.warning(f"Could not generate notebook: {e}")
    
    user_progress["started_projects"][project_id] = {
        "started_at": datetime.now().isoformat(),
        "current_step": 0,
        "completed_steps": [],
        "notebook_path": notebook_path
    }
    save_user_projects(user_progress)
    
    # Award XP for starting
    award_project_xp("project_started", 25)


def complete_project(project_id: str, user_progress: Dict):
    """Mark a project as completed."""
    if "completed_projects" not in user_progress:
        user_progress["completed_projects"] = []
    
    if project_id not in user_progress["completed_projects"]:
        user_progress["completed_projects"].append(project_id)
    
    if project_id in user_progress.get("started_projects", {}):
        user_progress["started_projects"][project_id]["completed_at"] = datetime.now().isoformat()
    
    save_user_projects(user_progress)
    
    # Award XP for completion
    award_project_xp("project_completed", 100)


def restart_project(project_id: str, user_progress: Dict):
    """Restart a completed project."""
    if project_id in user_progress.get("completed_projects", []):
        user_progress["completed_projects"].remove(project_id)
    
    user_progress["started_projects"][project_id] = {
        "started_at": datetime.now().isoformat(),
        "current_step": 0,
        "completed_steps": [],
        "restart_count": user_progress.get("started_projects", {}).get(project_id, {}).get("restart_count", 0) + 1
    }
    save_user_projects(user_progress)


def award_project_xp(activity: str, xp_amount: int):
    """Award XP for project activities."""
    try:
        from .progress_interface import load_progress, save_progress
        progress = load_progress()
        
        # Add XP
        progress['user_info']['xp'] = progress['user_info'].get('xp', 0) + xp_amount
        
        # Check for level up
        new_level = progress['user_info']['xp'] // 100 + 1
        if new_level > progress['user_info'].get('level', 1):
            progress['user_info']['level'] = new_level
        
        # Log activity
        if 'activities' not in progress:
            progress['activities'] = []
        
        progress['activities'].insert(0, {
            'name': activity.replace('_', ' ').title(),
            'xp': xp_amount,
            'date': datetime.now().isoformat()
        })
        
        save_progress(progress)
    except:
        pass  # Progress tracking is optional


def render_project_detail(project_id: str):
    """Render detailed view of a project."""
    registry = get_registry()
    project = registry.get_project_by_id(project_id)
    
    if not project:
        st.error("Project not found")
        return
    
    user_progress = load_user_projects()
    project_data = user_progress.get("started_projects", {}).get(project_id, {})
    is_completed = project_id in user_progress.get("completed_projects", [])
    
    # Back button
    if st.button("← Back to Projects"):
        st.session_state.show_project_detail = False
        st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Project header
    domain_icon = get_project_icon(project.domain)
    domain_color = get_domain_color(project.domain)
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {domain_color}20 0%, #1E293B 100%);
        border: 1px solid {domain_color};
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 1.5rem;
    ">
        <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
            <span style="font-size: 2.5rem;">{domain_icon}</span>
            <div>
                <h1 style="color: #F8FAFC; font-size: 1.5rem; margin: 0; font-weight: 700;">
                    {project.name}
                </h1>
                <div style="margin-top: 0.5rem;">
                    {get_difficulty_badge(project.difficulty)}
                    <span style="color: #94A3B8; margin-left: 0.5rem; font-size: 0.85rem;">
                        {project.domain.value.replace('_', ' ').title()} Domain
                    </span>
                </div>
            </div>
        </div>
        <p style="color: #CBD5E1; font-size: 1rem; margin: 0; line-height: 1.6;">
            {project.description}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Two columns layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Learning Objectives
        st.markdown("""
        <h3 style="color: #F8FAFC; margin: 1rem 0; font-size: 1.1rem; font-weight: 600;">
            📚 Learning Objectives
        </h3>
        """, unsafe_allow_html=True)
        
        for i, objective in enumerate(project.expected_outcomes, 1):
            completed = i <= len(project_data.get("completed_steps", []))
            icon = "✅" if completed else "⬜"
            color = "#22C55E" if completed else "#CBD5E1"
            
            st.markdown(f"""
            <div style="
                background: #1E293B;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 0.75rem 1rem;
                margin-bottom: 0.5rem;
                display: flex;
                align-items: center;
                gap: 0.75rem;
            ">
                <span style="font-size: 1.1rem;">{icon}</span>
                <span style="color: {color}; font-size: 0.9rem;">{objective}</span>
            </div>
            """, unsafe_allow_html=True)
        
        # Dataset info
        st.markdown("""
        <h3 style="color: #F8FAFC; margin: 1.5rem 0 1rem 0; font-size: 1.1rem; font-weight: 600;">
            📁 Dataset
        </h3>
        """, unsafe_allow_html=True)
        
        dataset_path = project.dataset_path
        st.markdown(f"""
        <div style="
            background: #1E293B;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 1rem;
        ">
            <code style="color: #14B8A6; font-size: 0.9rem;">{dataset_path}</code>
            <p style="color: #94A3B8; font-size: 0.8rem; margin: 0.5rem 0 0 0;">
                This dataset contains real-world data challenges you'll learn to solve.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Progress card
        if project_data:
            started_at = project_data.get("started_at", "")[:10]
            completed_steps = len(project_data.get("completed_steps", []))
            total_steps = len(project.expected_outcomes)
            progress_pct = int((completed_steps / total_steps * 100)) if total_steps > 0 else 0
            
            st.markdown(f"""
            <div style="
                background: #1E293B;
                border: 1px solid #334155;
                border-radius: 12px;
                padding: 1.25rem;
            ">
                <h4 style="color: #F8FAFC; margin: 0 0 1rem 0; font-size: 1rem; font-weight: 600;">
                    📊 Your Progress
                </h4>
                <div style="margin-bottom: 1rem;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                        <span style="color: #94A3B8; font-size: 0.8rem;">Progress</span>
                        <span style="color: #F8FAFC; font-size: 0.8rem; font-weight: 600;">{progress_pct}%</span>
                    </div>
                    <div style="background: #334155; border-radius: 4px; height: 8px; overflow: hidden;">
                        <div style="background: {domain_color}; height: 100%; width: {progress_pct}%; border-radius: 4px;"></div>
                    </div>
                </div>
                <div style="color: #94A3B8; font-size: 0.8rem;">
                    <div style="margin-bottom: 0.5rem;">
                        <strong style="color: #CBD5E1;">Started:</strong> {started_at}
                    </div>
                    <div>
                        <strong style="color: #CBD5E1;">Steps:</strong> {completed_steps} / {total_steps}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Quick Actions
        st.markdown("""
        <h4 style="color: #F8FAFC; margin: 1.5rem 0 1rem 0; font-size: 1rem; font-weight: 600;">
            ⚡ Quick Actions
        </h4>
        """, unsafe_allow_html=True)
        
        # Show notebook if already generated
        notebook_path = project_data.get("notebook_path") if project_data else None
        if notebook_path and os.path.exists(notebook_path):
            st.markdown(f"""
            <div style="
                background: #14532d;
                border: 1px solid #22C55E;
                border-radius: 8px;
                padding: 0.75rem;
                margin-bottom: 1rem;
            ">
                <div style="color: #22C55E; font-size: 0.85rem; font-weight: 600;">
                    📓 Notebook Ready!
                </div>
                <code style="color: #86EFAC; font-size: 0.75rem; word-break: break-all;">
                    {notebook_path}
                </code>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("📂 Open Notebook Location", key="open_nb_loc"):
                folder = os.path.dirname(os.path.abspath(notebook_path))
                st.info(f"Open this folder in VS Code or Jupyter:\n`{folder}`")
        else:
            # Generate notebook button
            if st.button("📓 Generate Notebook", key="gen_notebook", type="primary"):
                try:
                    nb_path = generate_project_notebook(project)
                    # Update project data with notebook path
                    if project_data:
                        project_data["notebook_path"] = nb_path
                        user_progress["started_projects"][project_id] = project_data
                        save_user_projects(user_progress)
                    st.success(f"✅ Notebook generated!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error generating notebook: {e}")
        
        # Load dataset button
        if st.button("📊 Explore Dataset", key="explore_data"):
            st.session_state.explore_dataset = project.dataset_path
            st.session_state.page = "Dataset Explorer"
            st.info("Navigate to Dataset Explorer to view this dataset!")
        
        # Mark step complete
        if project_data and not is_completed:
            st.markdown("<br>", unsafe_allow_html=True)
            current_step = len(project_data.get("completed_steps", []))
            if current_step < len(project.expected_outcomes):
                if st.button(f"✅ Complete Step {current_step + 1}", key="complete_step"):
                    if "completed_steps" not in project_data:
                        project_data["completed_steps"] = []
                    project_data["completed_steps"].append(current_step + 1)
                    user_progress["started_projects"][project_id] = project_data
                    save_user_projects(user_progress)
                    award_project_xp("step_completed", 15)
                    st.rerun()
    
    # Validation Rules section
    if project.validation_rules:
        st.markdown("""
        <h3 style="color: #F8FAFC; margin: 2rem 0 1rem 0; font-size: 1.1rem; font-weight: 600;">
            ✓ Validation Rules
        </h3>
        <p style="color: #94A3B8; font-size: 0.9rem; margin-bottom: 1rem;">
            Your cleaned data should pass these quality checks:
        </p>
        """, unsafe_allow_html=True)
        
        cols = st.columns(2)
        for i, rule in enumerate(project.validation_rules):
            with cols[i % 2]:
                st.markdown(f"""
                <div style="
                    background: #1E293B;
                    border: 1px solid #334155;
                    border-radius: 8px;
                    padding: 0.75rem 1rem;
                    margin-bottom: 0.5rem;
                ">
                    <div style="color: #F8FAFC; font-size: 0.9rem; font-weight: 500;">
                        {rule.name.replace('_', ' ').title()}
                    </div>
                    <div style="color: #94A3B8; font-size: 0.8rem;">
                        {rule.description}
                    </div>
                </div>
                """, unsafe_allow_html=True)


def render_projects_page():
    """Main entry point for the projects page."""
    
    # Check if we should show detail view
    if st.session_state.get('show_project_detail') and st.session_state.get('active_project'):
        render_project_detail(st.session_state.active_project)
    else:
        render_learning_projects()
