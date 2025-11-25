# manual_workflow.py

from datadojo import create_dojo
from datadojo.dojo_api import Domain, DifficultyLevel

# 1. Initialize DataDojo
print("1. Initializing DataDojo...")
dojo = create_dojo(educational_mode=True)
print("DataDojo initialized.\n")

# 2. List available projects
print("2. Listing available projects (beginner, e-commerce):")
projects = dojo.list_projects(domain=Domain.ECOMMERCE, difficulty=DifficultyLevel.BEGINNER)
if projects:
    print(f"Found {len(projects)} project(s):")
    for p in projects:
        print(f"  - ID: {p.id}, Name: {p.name}, Difficulty: {p.difficulty.value}")
    selected_project_id = projects[0].id
else:
    print("No beginner e-commerce projects found. Exiting.")
    exit()
print(f"\nSelecting project: {selected_project_id}\n")

# 3. Start a project
student_id = "manual_user"
print(f"3. Starting project '{selected_project_id}' for student '{student_id}'...")
project_instance = dojo.load_project(selected_project_id)
print(f"Project '{project_instance.info.name}' started successfully.")
print(f"Project Description: {project_instance.info.description}\n")

# 4. Learn a data science concept
print("4. Getting explanation for 'missing_values' concept:")
educational_interface = dojo.get_educational_interface()
concept = educational_interface.get_concept_explanation("missing_values")
print(f"Concept: {concept['title']}\nExplanation: {concept['explanation'][:200]}...\n")

# Simulate some work to record progress
print("Simulating some progress...")
project_instance.track_progress(student_id, "initial_data_load", ["data_types"])
print("Simulated completing a step and learning a concept.\n")

# 5. Check your progress
print(f"5. Checking progress for student '{student_id}' on project '{selected_project_id}':")
current_progress = project_instance.get_progress(student_id)
print(f"  - Completed Steps: {len(current_progress['completed_steps'])}")
print(f"  - Learned Concepts: {len(current_progress['learned_concepts'])}")
print(f"  - Average Skill Score: {current_progress['average_skill_score']:.1f}%")
print(f"  - Last Activity: {current_progress['last_activity']}\n")

print("Manual DataDojo workflow demonstration complete!")
