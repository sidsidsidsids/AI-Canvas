package com.siliconvalley.domain.item.stage.dto;

import com.siliconvalley.domain.item.stage.domain.Stage;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Getter
@NoArgsConstructor(access = AccessLevel.PRIVATE)
public class StagePostSuccessResponse {

    private Long id;

    public StagePostSuccessResponse(Stage stage) {
        this.id = stage.getId();
    }
}
