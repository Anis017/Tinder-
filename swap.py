import random
import math
import time
from typing import List, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class User:
    id: int
    name: str
    age: int
    location: Tuple[float, float]  # (lat, lon)
    elo_rating: float = 1500.0
    activity_score: float = 1.0
    last_active: datetime = None
    swipe_history: List[int] = None
    match_history: List[int] = None
    
    def __post_init__(self):
        if self.swipe_history is None:
            self.swipe_history = []
        if self.match_history is None:
            self.match_history = []
        if self.last_active is None:
            self.last_active = datetime.now()

class DatingAlgorithm:
    def __init__(self, k_factor: float = 32.0):
        self.k_factor = k_factor  # Elo sensitivity
        self.users: Dict[int, User] = {}
        self.swipe_data: List[Dict] = []
        
    def add_user(self, user: User):
        """Add a user to the system"""
        self.users[user.id] = user
    
    def calculate_distance(self, user1: User, user2: User) -> float:
        """Calculate distance between two users using Haversine formula"""
        lat1, lon1 = user1.location
        lat2, lon2 = user2.location
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371  # Earth's radius in kilometers
        
        return c * r
    
    def expected_score(self, rating1: float, rating2: float) -> float:
        """Calculate expected match probability using Elo formula"""
        return 1 / (1 + 10**((rating2 - rating1) / 400))
    
    def update_elo_ratings(self, user1_id: int, user2_id: int, user1_swiped_right: bool, user2_swiped_right: bool):
        """Update Elo ratings based on swipe results (FaceMash style)"""
        user1 = self.users[user1_id]
        user2 = self.users[user2_id]
        
        # Calculate expected scores
        expected1 = self.expected_score(user1.elo_rating, user2.elo_rating)
        expected2 = self.expected_score(user2.elo_rating, user1.elo_rating)
        
        # Determine actual scores based on mutual attraction
        if user1_swiped_right and user2_swiped_right:
            # Mutual match - both get positive reinforcement
            actual1, actual2 = 1.0, 1.0
        elif user1_swiped_right and not user2_swiped_right:
            # User1 liked User2 but not reciprocated
            actual1, actual2 = 0.0, 1.0
        elif not user1_swiped_right and user2_swiped_right:
            # User2 liked User1 but not reciprocated
            actual1, actual2 = 1.0, 0.0
        else:
            # Neither liked each other - neutral
            actual1, actual2 = 0.5, 0.5
        
        # Update ratings
        user1.elo_rating += self.k_factor * (actual1 - expected1)
        user2.elo_rating += self.k_factor * (actual2 - expected2)
        
        # Keep ratings positive
        user1.elo_rating = max(user1.elo_rating, 100)
        user2.elo_rating = max(user2.elo_rating, 100)
    
    def calculate_recency_factor(self, user: User) -> float:
        """Calculate recency bonus based on last activity"""
        hours_since_active = (datetime.now() - user.last_active).total_seconds() / 3600
        return math.exp(-0.1 * hours_since_active)  # Decay factor
    
    def calculate_distance_penalty(self, distance: float) -> float:
        """Calculate distance penalty (closer is better)"""
        return math.exp(-0.01 * distance**2)
    
    def calculate_compatibility_score(self, user1: User, user2: User) -> float:
        """Simple compatibility based on age difference"""
        age_diff = abs(user1.age - user2.age)
        return math.exp(-0.1 * age_diff)
    
    def calculate_profile_score(self, viewer: User, candidate: User) -> float:
        """Calculate overall score for showing candidate to viewer"""
        if candidate.id == viewer.id:
            return 0
        
        # Component weights
        w_elo = 0.4
        w_recency = 0.2
        w_distance = 0.2
        w_compatibility = 0.2
        
        # Calculate components
        elo_component = candidate.elo_rating / 2000.0  # Normalize
        recency_component = self.calculate_recency_factor(candidate)
        distance = self.calculate_distance(viewer, candidate)
        distance_component = self.calculate_distance_penalty(distance)
        compatibility_component = self.calculate_compatibility_score(viewer, candidate)
        
        # Weighted sum
        score = (w_elo * elo_component + 
                w_recency * recency_component + 
                w_distance * distance_component + 
                w_compatibility * compatibility_component)
        
        return score
    
    def get_recommendation_stack(self, user_id: int, stack_size: int = 10) -> List[User]:
        """Get ordered list of potential matches for a user"""
        viewer = self.users[user_id]
        candidates = []
        
        for candidate_id, candidate in self.users.items():
            if candidate_id == user_id:
                continue
            if candidate_id in viewer.swipe_history:
                continue  # Already swiped on this person
                
            score = self.calculate_profile_score(viewer, candidate)
            candidates.append((score, candidate))
        
        # Sort by score (highest first) and return top candidates
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [candidate for _, candidate in candidates[:stack_size]]
    
    def simulate_swipe(self, swiper_id: int, target_id: int, swipe_right: bool):
        """Simulate a swipe and update system state"""
        swiper = self.users[swiper_id]
        target = self.users[target_id]
        
        # Record swipe
        swiper.swipe_history.append(target_id)
        swiper.last_active = datetime.now()
        
        # Check if target has already swiped on swiper
        target_swiped_right = swiper_id in target.swipe_history and swiper_id in target.match_history
        
        # Update Elo ratings
        self.update_elo_ratings(swiper_id, target_id, swipe_right, target_swiped_right)
        
        # Record match if mutual
        is_match = swipe_right and target_swiped_right
        if is_match:
            swiper.match_history.append(target_id)
            target.match_history.append(swiper_id)
        
        # Store swipe data for analysis
        self.swipe_data.append({
            'swiper_id': swiper_id,
            'target_id': target_id,
            'swipe_right': swipe_right,
            'is_match': is_match,
            'timestamp': datetime.now()
        })
        
        return is_match
    
    def get_user_stats(self, user_id: int) -> Dict:
        """Get statistics for a user"""
        user = self.users[user_id]
        total_swipes = len(user.swipe_history)
        matches = len(user.match_history)
        match_rate = matches / total_swipes if total_swipes > 0 else 0
        
        return {
            'user_id': user_id,
            'name': user.name,
            'elo_rating': round(user.elo_rating, 1),
            'total_swipes': total_swipes,
            'matches': matches,
            'match_rate': round(match_rate * 100, 1),
            'activity_score': round(user.activity_score, 2)
        }

# Demo usage and simulation
def run_simulation():
    """Run a simulation of the dating algorithm"""
    algorithm = DatingAlgorithm()
    
    # Create sample users
    users = [
        User(1, "Alice", 25, (40.7128, -74.0060)),  # NYC
        User(2, "Bob", 28, (40.7589, -73.9851)),   # NYC
        User(3, "Charlie", 30, (40.6782, -73.9442)), # NYC
        User(4, "Diana", 26, (40.7505, -73.9934)),   # NYC
        User(5, "Eve", 24, (40.7282, -73.7949)),     # NYC
        User(6, "Frank", 32, (40.6892, -74.0445)),   # NYC
    ]
    
    # Add users to system
    for user in users:
        algorithm.add_user(user)
    
    print("=== Dating Algorithm Simulation ===\n")
    
    # Simulate random swipes
    print("Simulating swipes...\n")
    for _ in range(50):
        swiper_id = random.choice(list(algorithm.users.keys()))
        
        # Get recommendation stack
        recommendations = algorithm.get_recommendation_stack(swiper_id, 5)
        if not recommendations:
            continue
            
        target = random.choice(recommendations)
        
        # Simulate swipe decision (70% chance of swiping right)
        swipe_right = random.random() < 0.7
        
        is_match = algorithm.simulate_swipe(swiper_id, target.id, swipe_right)
        
        if is_match:
            print(f"🎉 MATCH! {algorithm.users[swiper_id].name} ↔ {target.name}")
    
    # Display final statistics
    print("\n=== Final User Statistics ===")
    for user_id in algorithm.users:
        stats = algorithm.get_user_stats(user_id)
        print(f"{stats['name']}: Elo={stats['elo_rating']}, "
              f"Matches={stats['matches']}, "
              f"Match Rate={stats['match_rate']}%")
    
    # Show recommendation stack for Alice
    print(f"\n=== Recommendation Stack for Alice ===")
    alice_recs = algorithm.get_recommendation_stack(1, 5)
    for i, user in enumerate(alice_recs, 1):
        score = algorithm.calculate_profile_score(algorithm.users[1], user)
        print(f"{i}. {user.name} (Age: {user.age}, Elo: {user.elo_rating:.1f}, Score: {score:.3f})")

if __name__ == "__main__":
    run_simulation()
