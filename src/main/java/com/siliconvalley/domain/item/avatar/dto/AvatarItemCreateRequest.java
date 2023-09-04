package com.siliconvalley.domain.item.avatar.dto;

import com.siliconvalley.domain.item.avatar.domain.Avatar;
import com.siliconvalley.domain.item.item.domain.Item;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.NoArgsConstructor;

import javax.validation.Valid;

@Getter
@NoArgsConstructor(access = AccessLevel.PACKAGE)
public class AvatarItemCreateRequest {

    @Valid
    private Long itemPrice;

    @Valid
    private String avatarName;

    @Valid
    private String avatarImage;

    AvatarItemCreateRequest(
            @Valid Long itemPrice,
            @Valid String avatarName,
            @Valid String avatarImage
    ) {
        this.itemPrice = itemPrice;
        this.avatarName = avatarName;
        this.avatarImage = avatarImage;
    }

    public Item toItemEntity() {
        return Item.builder()
                .itemPrice(itemPrice)
                .build();
    }

    public Avatar toAvatarEntity() {
        return Avatar.builder()
                .avatarName(avatarName)
                .avatarImage(avatarImage)
                .build();
    }
}
