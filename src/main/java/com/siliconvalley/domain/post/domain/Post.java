package com.siliconvalley.domain.post.domain;

import com.siliconvalley.domain.canvas.domain.Canvas;
import com.siliconvalley.domain.profile.domain.Profile;
import lombok.AccessLevel;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.hibernate.annotations.CreationTimestamp;

import javax.persistence.*;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@Slf4j
@Entity
@Table(name = "post")
public class Post {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "Id")
    private Long Id;

    @CreationTimestamp
    @Column(name = "created_at", nullable = false, updatable = false)
    private LocalDateTime createdAt;

    @ManyToOne
    @JoinColumn(name = "profile_id")
    private Profile profile;

    @OneToOne
    @JoinColumn(name = "canvas_id")
    private Canvas canvas;

    @OneToMany(mappedBy = "post", cascade = CascadeType.PERSIST, orphanRemoval = true, fetch = FetchType.LAZY)
    private List<Emotion> emotions = new ArrayList<>();

    @Builder
    public Post(Profile profile, Canvas canvas) {
        this.profile = profile;
        this.canvas = canvas;
    }

    public Emotion buildEmotion(EmotionType emotionType, Profile profile){
        return Emotion.builder()
                .post(this)
                .profile(profile)
                .emotionType(emotionType)
                .build();
    }

    public void addEmotion(EmotionType emotionType, Profile profile){
        this.emotions.add(buildEmotion(emotionType, profile));
    }

}
